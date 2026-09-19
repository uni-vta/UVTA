import concurrent.futures
import os.path as osp
from collections import defaultdict

import cv2
import numpy as np
import zarr
from uvta.common.imagecodecs_numcodecs import register_codecs
from uvta.diffusion_policy.dataloader.ring_overlay import (
    VALID_EMBODIMENTS,
    canonical_embodiment,
    parse_overlay_cfg,
)
from tqdm import tqdm

register_codecs()


def sorted_episode_keys(root):
    """Episode group names in NUMERIC order (``episode_0, episode_1, ...``).

    ``zarr``'s ``group_keys()`` is alphabetical, which orders episodes as
    ``episode_0, episode_1, episode_10, episode_100, ...``.  That makes any
    "first N episodes" subset (``max_episode``) a scattered, non-obvious slice --
    bad for reproducibility and, for ablations on dataset SIZE, actively
    misleading.  Sorting on the trailing integer makes "first N" mean episodes
    ``0 .. N-1``, i.e. a contiguous prefix of the recording session.
    """
    keys = list(root.group_keys())

    def _key(name):
        digits = "".join(ch for ch in str(name).rsplit("_", 1)[-1] if ch.isdigit())
        return (0, int(digits)) if digits else (1, str(name))

    return sorted(keys, key=_key)


def resolve_episode_limits(max_episode, data_path):
    """Split ``max_episode`` into a GLOBAL cap and PER-DATASET caps.

    ``max_episode`` may be:
      * ``None``  -> no limit anywhere;
      * an ``int`` -> a GLOBAL cap across all ``data_dirs`` (legacy behaviour,
        used by the trainers' ``debug`` mode);
      * a ``list`` / ``tuple`` aligned to ``data_dirs`` -> a PER-DATASET cap, for
        ablations that vary how much of each dataset is fed to the policy (e.g.
        ``[50, null]`` = 50 robot episodes + all human episodes).  ``null`` /
        negative entries mean "no limit" for that dataset.

    Returns ``(global_cap, per_dir_caps)`` where exactly one is non-None.
    """
    if max_episode is None:
        return None, None
    # Duck-type the sequence check: hydra hands this over as an OmegaConf
    # ``ListConfig``, which is NOT a list/tuple instance.
    is_seq = not isinstance(max_episode, (int, float, str, bytes)) and hasattr(
        max_episode, "__iter__"
    )
    if is_seq:
        max_episode = list(max_episode)
        n = len(data_path)
        if len(max_episode) != n:
            raise ValueError(
                f"max_episode has {len(max_episode)} entries but there are {n} "
                f"data_dirs; give one limit per dataset (use null for 'all')."
            )
        caps = [
            None if (m is None or int(m) < 0) else int(m) for m in max_episode
        ]
        return None, caps
    return int(max_episode), None


class ReplayBuffer:
    def __init__(
        self,
        data_path,
        load_camera_ids=[],
        camera_resize_shape=[],
        max_episode=None,
        max_workers=4,
        bgr2rgb=False,  # Add new parameter
    ) -> None:
        self.data_path = data_path
        self.load_camera_ids = load_camera_ids
        self.camera_resize_shape = camera_resize_shape
        self.max_workers = max_workers
        # ``max_episode``: int -> global cap; list -> per-dataset caps.  See
        # ``resolve_episode_limits``.
        self.max_episode, self.max_episode_per_dir = resolve_episode_limits(
            max_episode, data_path
        )
        self.bgr2rgb = bgr2rgb  # Store parameter
        self.initiate_memory_buffer()
        self.load_data_to_memory()

    def _episode_cap(self, path_idx):
        """Per-dataset episode cap for ``data_path[path_idx]`` (None = all)."""
        if self.max_episode_per_dir is None:
            return None
        return self.max_episode_per_dir[path_idx]

    def initiate_memory_buffer(self):
        self.memory_buffer = defaultdict(list)

    def load_data_to_memory(self):
        load_episode_num = 0
        for path_idx, path in enumerate(self.data_path):
            root = zarr.open(path, mode="r")
            episodes = sorted_episode_keys(root)
            cap = self._episode_cap(path_idx)
            n_this_path = 0
            for episode in episodes:
                if cap is not None and n_this_path >= cap:
                    break
                self.memory_buffer["action"].append(
                    self.load_low_dim_data(root, osp.join(episode, "action"))
                )
                self.memory_buffer["proprioception"].append(
                    self.load_low_dim_data(root, osp.join(episode, "proprioception"))
                )
                for camera_ids in self.load_camera_ids:
                    cam_name = f"camera_{camera_ids}"
                    self.memory_buffer[cam_name].append(
                        self.load_visual_data(root, osp.join(episode, cam_name, "rgb"))
                    )
                load_episode_num += 1
                n_this_path += 1
                if (
                    self.max_episode is not None
                    and load_episode_num >= self.max_episode
                ):
                    break

        self.eps_end = np.cumsum([len(x) for x in self.memory_buffer["action"]])
        for k, v in self.memory_buffer.items():
            self.memory_buffer[k] = np.concatenate(v)

    def load_low_dim_data(self, root, low_dim_path):
        return root[low_dim_path][:].astype(np.float32)

    def load_visual_data(self, root, visual_path, dim=3):
        visual_shape = root[visual_path].shape
        np_arr_shape = (
            (visual_shape[0], *self.camera_resize_shape, dim)
            if self.camera_resize_shape
            else visual_shape
        )
        np_arr = np.zeros(np_arr_shape, dtype=np.uint8)

        def load_img(zarr_arr, visual_path, index, np_arr):
            try:
                img = zarr_arr[visual_path][index]
                if self.camera_resize_shape:
                    img = cv2.resize(img, self.camera_resize_shape)
                if self.bgr2rgb:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                np_arr[index] = img
                return True
            except Exception as e:
                print(e)
                return False

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = set()
            for i in range(visual_shape[0]):
                futures.add(executor.submit(load_img, root, visual_path, i, np_arr))

            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                if not f.result():
                    raise RuntimeError("Failed to load image!")

        return np_arr

    def __repr__(self) -> str:
        rep = ""
        for k, v in self.memory_buffer.items():
            rep += f"{k}, {v.shape}\n"
        rep += f"eps_end, {self.eps_end}\n"
        return rep

    def __getitem__(self, key):
        return self.memory_buffer[key]

    def remove_key(self, key):
        del self.memory_buffer[key]


class UVTAReplayBuffer(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        # Extract optional parameters
        self.skip_proprioception = kwargs.pop("skip_proprioception", False)
        # When True, load the per-episode fingertip_pose_wrist (T, 5, 6)
        # **state** array (FK(proprioception[t])) into
        # ``memory_buffer["fingertip_pose_wrist"]`` for fingertip-style
        # proprio (observation).
        self.load_fingertip = kwargs.pop("load_fingertip", False)
        # When True, load the per-episode fingertip_action (T, 5, 6)
        # **next-state action** array (FK(proprioception[t+1]), with the
        # last frame repeating FK(proprioception[T-1])) into
        # ``memory_buffer["fingertip_action"]`` for fingertip-style
        # action targets.  This is independent of ``load_fingertip`` so
        # you can use one without the other.
        self.load_fingertip_action = kwargs.pop("load_fingertip_action", False)
        # The per-episode ``pose_action`` (T, 6) wrist action stream (xyz +
        # axis-angle rotvec = the commanded next-state target, shift +1 from the
        # observed ``pose`` state) is loaded automatically whenever the field is
        # present in the data (auto-detected in ``_preallocate_arrays``).
        # UVTADataset turns it into the relative action target on the fly
        # (anchored on the current ``pose`` state).  No flag needed.
        self.has_pose_action = False

        # ``action_from_next_state``: use the OBSERVED NEXT STATE as the action
        # target instead of the recorded command streams.  When True we SYNTHESIZE
        #   pose_action[t]  := pose[t+1]           (wrist action = next state)
        #   hand_action[t]  := proprioception[t+1] (joint action = next state)
        # per episode (with the last frame repeating, matching the shift-+1
        # convention).  This overrides whatever ``pose_action`` / ``hand_action``
        # the zarr carries, so downstream normalization, UVTADataset (which
        # relativizes pose_action onto the current pose[t] state) and the deploy
        # side all work UNCHANGED -- the only difference is what the target
        # numerically contains.  Requires the ``proprioception`` stream (we force
        # skip_proprioception=False in UVTADataset when this is on).
        self.action_from_next_state = bool(
            kwargs.pop("action_from_next_state", False)
        )
        if self.action_from_next_state:
            print(
                "[UVTAReplayBuffer] action_from_next_state=True -> "
                "synthesizing pose_action:=pose[t+1] and "
                "hand_action:=proprioception[t+1] (observed next state as "
                "action; last frame repeats)."
            )

        # ``synthesize_next_state``: expose the observed next state as its OWN
        # streams instead of overwriting the command streams:
        #   pose_next[t]  := pose[t+1]            (wrist state at t+1)
        #   joint_next[t] := proprioception[t+1]  (joint state at t+1)
        # This is what lets a policy predict the command AND the resulting state
        # as two separate output blocks; ``action_from_next_state`` above is the
        # older either/or form, which destroys the command stream.  The two are
        # independent and may be combined, though there is no reason to.
        self.synthesize_next_state = bool(
            kwargs.pop("synthesize_next_state", False)
        )
        if self.synthesize_next_state:
            print(
                "[UVTAReplayBuffer] synthesize_next_state=True -> adding "
                "pose_next:=pose[t+1] and joint_next:=proprioception[t+1] as "
                "separate streams (last frame repeats); the recorded "
                "pose_action / hand_action are left untouched."
            )

        # --- ring overlay -----------------------------------------------
        # ``data_dirs_embodiment``: optional list[str] with one element per
        # entry in ``data_path`` describing the embodiment that produced
        # the data ("teleop", "exoskeleton", "manus").  When unset every
        # path defaults to "teleop" (= no overlay applied).
        # ``ring_overlay``: dict-like config; see ring_overlay.py.
        embodiment_list = kwargs.pop("data_dirs_embodiment", None)
        overlay_cfg = kwargs.pop("ring_overlay", None)
        self._raw_embodiment_list = embodiment_list
        self._ring_overlay, self._ring_embodiments = parse_overlay_cfg(overlay_cfg)
        # Mark for ``load_data_to_memory`` whether per-episode overlay is on.
        self._ring_overlay_enabled = self._ring_overlay is not None

        # Build the per-path embodiment list NOW (before super().__init__
        # which will internally call load_data_to_memory).  We pull
        # ``data_path`` from args/kwargs since super hasn't set ``self.data_path``
        # yet.
        if "data_path" in kwargs:
            _data_path = kwargs["data_path"]
        elif args:
            _data_path = args[0]
        else:
            raise TypeError(
                "UVTAReplayBuffer requires data_path as the first positional "
                "argument or as the ``data_path`` keyword."
            )
        self.embodiments = self._build_embodiment_list(_data_path)

        # Handle FSR / tactile options
        #
        # ``enable_fsr``      : master switch for loading a per-frame tactile
        #                       stream into ``memory_buffer[fsr_source_key]``.
        #                       The stream is stored (and later normalized /
        #                       saved to stats.pickle / carried in the batch)
        #                       under its SOURCE key so every consumer refers
        #                       to it by its real name (e.g. "fsr" / "force").
        # ``fsr_source_key``  : which per-episode zarr field to read the tactile
        #                       stream from.  Defaults to ``"fsr"`` (legacy).
        #                       Set to e.g. ``"force"`` to co-train on the 5-D
        #                       finger-force stream instead of the 100-D FSR.
        # ``fsr_binarize``    : when True (legacy default) threshold the raw
        #                       tactile reading against ``fsr_binary_cutoff`` to
        #                       produce a 0/1 contact signal.  When False the
        #                       raw (float) values are kept and later
        #                       range-normalized like any other stream (use this
        #                       for the continuous ``force`` field).
        # ``fsr_binary_cutoff``: threshold(s) used only when ``fsr_binarize``.
        #                       Either a scalar / length-1 list (broadcast over
        #                       all channels) or a list matching the tactile
        #                       feature dim.
        # ``fsr_baseline_correct``: per-episode baseline calibration.  Some
        #                       sensors (notably the HUMAN capture) carry a
        #                       per-take DC offset, so the "resting" reading is
        #                       not zero.  When enabled, for each episode whose
        #                       embodiment is in ``fsr_baseline_embodiments`` we
        #                       subtract the mean of the first
        #                       ``fsr_baseline_frames`` frames from the WHOLE
        #                       episode BEFORE binarize / normalize, so the
        #                       stream represents the true delta above rest.
        # ``fsr_baseline_frames``: number of leading frames averaged for the
        #                       per-episode baseline (default 5).
        # ``fsr_baseline_embodiments``: which embodiments get the MEAN-of-first-N
        #                       correction (default ["human"]).
        # ``fsr_baseline_first_frame_embodiments``: which embodiments get a
        #                       FIRST-FRAME baseline instead (subtract frame 0,
        #                       i.e. N=1 regardless of ``fsr_baseline_frames``).
        #                       Use for the ROBOT stream, whose per-session DC
        #                       offset is best captured by the very first frame.
        #                       Takes precedence if an embodiment is in both
        #                       lists.  NOTE: whatever correction is applied here
        #                       must be replicated at DEPLOY on the live stream
        #                       (subtract the session's first-frame baseline),
        #                       or train/deploy will mismatch.
        self.enable_fsr = False
        self.fsr_source_key = "fsr"
        self.fsr_binarize = True
        self.fsr_binary_cutoff = None
        self.fsr_baseline_correct = False
        self.fsr_baseline_frames = 5
        self.fsr_baseline_embodiments = set()
        self.fsr_baseline_first_frame_embodiments = set()
        if kwargs.get("enable_fsr", False):
            self.enable_fsr = kwargs.pop("enable_fsr")
            self.fsr_source_key = str(kwargs.pop("fsr_source_key", "fsr"))
            self.fsr_binarize = bool(kwargs.pop("fsr_binarize", True))
            cutoff = kwargs.pop("fsr_binary_cutoff", None)
            if self.fsr_binarize:
                if cutoff is None:
                    raise ValueError(
                        "enable_fsr with fsr_binarize=True requires "
                        "fsr_binary_cutoff to be set."
                    )
                self.fsr_binary_cutoff = np.array(cutoff, dtype=np.float32)
            self.fsr_baseline_correct = bool(
                kwargs.pop("fsr_baseline_correct", False)
            )
            self.fsr_baseline_frames = int(kwargs.pop("fsr_baseline_frames", 5))
            _bl_emb = kwargs.pop("fsr_baseline_embodiments", ["human"])
            self.fsr_baseline_embodiments = set(
                canonical_embodiment(e) for e in _bl_emb
            )
            _bl_ff = kwargs.pop("fsr_baseline_first_frame_embodiments", [])
            self.fsr_baseline_first_frame_embodiments = set(
                canonical_embodiment(e) for e in _bl_ff
            )
            print("===============")
            print(
                f"enable_fsr from field '{self.fsr_source_key}', "
                f"binarize={self.fsr_binarize}, "
                f"binary_cutoff={self.fsr_binary_cutoff}"
            )
            if self.fsr_baseline_correct:
                print(
                    f"tactile baseline correction: subtract mean of first "
                    f"{self.fsr_baseline_frames} frames per episode for "
                    f"embodiments {sorted(self.fsr_baseline_embodiments)}"
                )
                if self.fsr_baseline_first_frame_embodiments:
                    print(
                        f"tactile baseline correction: subtract the FIRST frame "
                        f"per episode for embodiments "
                        f"{sorted(self.fsr_baseline_first_frame_embodiments)}"
                    )
            print("===============")
        else:
            # Clean up kwargs if keys exist
            kwargs.pop("enable_fsr", None)
            kwargs.pop("fsr_binary_cutoff", None)
            kwargs.pop("fsr_source_key", None)
            kwargs.pop("fsr_binarize", None)
            kwargs.pop("fsr_baseline_correct", None)
            kwargs.pop("fsr_baseline_frames", None)
            kwargs.pop("fsr_baseline_embodiments", None)
            kwargs.pop("fsr_baseline_first_frame_embodiments", None)

        # --- bimanual arm prefixes --------------------------------------
        # ``arm_prefixes``: list of per-arm field prefixes.  Single-arm data
        # stores the per-arm streams under bare names (``pose`` / ``hand_action``
        # / ...), so the default ``[""]`` reproduces the legacy behaviour
        # exactly.  Dual-arm data (e.g. ``chips_teleop``) stores them under
        # ``left_*`` / ``right_*``, so pass ``arm_prefixes=["left_", "right_"]``.
        # Every per-arm stream is loaded into a buffer key ``f"{prefix}{base}"``
        # (so single-arm keys stay ``pose`` etc.), while the cameras are shared
        # across arms and stay ``camera_{id}``.  The dataset later builds the
        # per-arm action / obs blocks from these prefixed keys.
        arm_prefixes = kwargs.pop("arm_prefixes", None)
        if arm_prefixes is None:
            arm_prefixes = [""]
        arm_prefixes = [str(p) for p in arm_prefixes]
        if len(arm_prefixes) == 0:
            raise ValueError("arm_prefixes must be a non-empty list (default [''])")
        self.arm_prefixes = arm_prefixes
        self.is_bimanual = len(self.arm_prefixes) > 1
        if self.is_bimanual:
            print(
                f"[UVTAReplayBuffer] bimanual mode: arm_prefixes="
                f"{self.arm_prefixes} (per-arm streams loaded under prefixed "
                "keys; cameras shared)."
            )

        # Initialize parent class
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Embodiment helper
    # ------------------------------------------------------------------

    def _build_embodiment_list(self, data_path) -> list[str]:
        """Return a list[str] of length ``len(data_path)``.

        If the user passed ``data_dirs_embodiment`` we use that; otherwise
        every entry defaults to ``"teleop"`` (which never triggers the
        ring overlay, matching the legacy behaviour).
        """
        if self._raw_embodiment_list is None:
            # Default embodiment when unspecified is ``robot`` (the real
            # capture that carries the camera ring and never triggers the
            # ring overlay), matching the legacy default (``teleop``).
            embs = ["robot"] * len(data_path)
        else:
            embs = list(self._raw_embodiment_list)
            if len(embs) != len(data_path):
                raise ValueError(
                    f"data_dirs_embodiment has length {len(embs)} but "
                    f"data_dirs has length {len(data_path)}; the two "
                    f"must be the same length."
                )
            # Canonicalize (accepts legacy teleop/exoskeleton/manus and the
            # new robot/human; raises on typos).
            embs = [canonical_embodiment(e) for e in embs]
        if self._ring_overlay_enabled:
            covered = [e for e in embs if e in self._ring_embodiments]
            print(
                f"[RingOverlay] enabled; will be applied to "
                f"{len(covered)}/{len(embs)} data_dirs whose embodiment is "
                f"in {sorted(self._ring_embodiments)}."
            )
        return embs

    def frame_embodiment_labels(self) -> np.ndarray:
        """``(total_frames,)`` object array: canonical embodiment per frame.

        Built from ``eps_end`` (per-episode cumulative frame count) and
        ``episode_embodiment`` (per-episode embodiment).  Frame ``f`` belongs
        to episode ``i`` iff ``eps_end[i-1] <= f < eps_end[i]``.
        """
        eps_end = np.asarray(self.eps_end)
        total = int(eps_end[-1]) if len(eps_end) else 0
        labels = np.empty(total, dtype=object)
        start = 0
        for i, end in enumerate(eps_end):
            labels[start:end] = self.episode_embodiment[i]
            start = int(end)
        return labels

    def _should_apply_ring(self, path_idx: int) -> bool:
        if not self._ring_overlay_enabled:
            return False
        return self.embodiments[path_idx] in self._ring_embodiments

    def load_data_to_memory(self):
        # Use a generator to avoid loading all episodes at once
        episode_lengths = []
        total_frames = 0

        # First pass: count total frames and get episode lengths.
        #
        # The episode SELECTION here must match the load pass below EXACTLY:
        # that pass indexes ``episode_lengths`` with a single flat counter, so
        # any mismatch would misalign every episode's slice (and leave the tail
        # of the pre-allocated buffer as unwritten zeros).  Both passes
        # therefore iterate ``data_path`` in order, take episodes in NUMERIC
        # order, and apply the same caps:
        #   * ``max_episode`` as an int   -> one GLOBAL cap across all datasets;
        #   * ``max_episode`` as a list   -> a PER-DATASET cap (ablations on how
        #     much of each dataset is used).
        scanned_num = 0
        self._episodes_per_path = []
        for path_idx, path in enumerate(self.data_path):
            root = zarr.open(path, mode="r")
            episodes = sorted_episode_keys(root)
            cap = self._episode_cap(path_idx)

            stop = False
            n_this_path = 0
            for episode in tqdm(episodes, desc="Scanning episodes"):
                if cap is not None and n_this_path >= cap:
                    break
                # Get episode length from pose data (first arm's pose stream;
                # all arms share the same episode length).
                pose_path = osp.join(episode, f"{self.arm_prefixes[0]}pose")
                episode_length = len(root[pose_path])
                episode_lengths.append(episode_length)
                total_frames += episode_length
                scanned_num += 1
                n_this_path += 1

                if self.max_episode is not None and scanned_num >= self.max_episode:
                    stop = True
                    break
            self._episodes_per_path.append(n_this_path)
            if stop:
                break
        while len(self._episodes_per_path) < len(self.data_path):
            self._episodes_per_path.append(0)
        print(
            "[replay_buffer] episodes per dataset: "
            + ", ".join(
                f"{osp.basename(str(p))}={n}"
                for p, n in zip(self.data_path, self._episodes_per_path)
            )
        )

        # Pre-allocate arrays with the exact size needed
        self._preallocate_arrays(total_frames)

        # Second pass: load the data
        load_episode_num = 0
        current_idx = 0

        # Per-episode provenance: which ``data_path`` (dataset / embodiment)
        # index each loaded episode came from.  Same length/order as
        # ``eps_end``.  Consumed by the co-training uniform sampler so it can
        # assign each training sample a per-dataset weight.
        self.episode_data_dir_idx = []
        # The zarr group name of each loaded episode ("episode_37", ...), same
        # length/order as ``eps_end``.  This is the only stable identity an
        # episode has: ``eps_end`` offsets shift as soon as ``data_dirs`` or
        # ``max_episode`` change, so anything joining EXTERNAL per-episode data
        # onto this buffer (e.g. a cached stage-1 rollout) must key off the name.
        self.episode_names = []

        for path_idx, path in enumerate(self.data_path):
            root = zarr.open(path, mode="r")
            # Same order + same cap as the scan pass above (see its comment).
            episodes = sorted_episode_keys(root)
            cap = self._episode_cap(path_idx)
            n_this_path = 0
            apply_ring = self._should_apply_ring(path_idx)

            for episode in tqdm(
                episodes,
                desc=(
                    f"Loading episodes [{self.embodiments[path_idx]}"
                    + (" +ring" if apply_ring else "")
                    + f"] {osp.basename(str(path))}"
                ),
            ):
                if cap is not None and n_this_path >= cap:
                    break
                n_this_path += 1
                episode_length = episode_lengths[load_episode_num]
                end_idx = current_idx + episode_length

                # Per-arm low-dim streams.  For single-arm data ``arm_prefixes``
                # is ``[""]`` so the keys stay ``hand_action`` / ``pose`` / ...
                # (unchanged); for dual-arm data they are ``left_*`` / ``right_*``.
                for prefix in self.arm_prefixes:
                    # Load hand action
                    self.memory_buffer[f"{prefix}hand_action"][
                        current_idx:end_idx
                    ] = self.load_low_dim_data(
                        root, osp.join(episode, f"{prefix}hand_action")
                    )

                    # Load pose (observed wrist STATE)
                    self.memory_buffer[f"{prefix}pose"][current_idx:end_idx] = (
                        self.load_low_dim_data(
                            root, osp.join(episode, f"{prefix}pose")
                        )
                    )

                    # Load pose_action (commanded wrist ACTION) when present in
                    # the data.  Skipped under ``action_from_next_state`` -- we
                    # synthesize it from the shifted ``pose`` state below instead.
                    if self.has_pose_action and not self.action_from_next_state:
                        self.memory_buffer[f"{prefix}pose_action"][
                            current_idx:end_idx
                        ] = self.load_low_dim_data(
                            root, osp.join(episode, f"{prefix}pose_action")
                        )

                    # Load proprioception if needed
                    if not self.skip_proprioception:
                        self.memory_buffer[f"{prefix}proprioception"][
                            current_idx:end_idx
                        ] = self.load_low_dim_data(
                            root, osp.join(episode, f"{prefix}proprioception")
                        )

                    # ``action_from_next_state``: overwrite the action streams
                    # with the OBSERVED NEXT STATE (shift +1, last frame
                    # repeating) so the target is where the hand/wrist actually
                    # goes next rather than the recorded command.  Done per
                    # episode on the just-loaded ``pose`` / ``proprioception``
                    # slices (per arm).
                    if self.action_from_next_state:
                        pose_ep = self.memory_buffer[f"{prefix}pose"][
                            current_idx:end_idx
                        ]
                        pose_next = pose_ep.copy()
                        if pose_next.shape[0] > 1:
                            pose_next[:-1] = pose_ep[1:]
                        self.memory_buffer[f"{prefix}pose_action"][
                            current_idx:end_idx
                        ] = pose_next
                        proprio_ep = self.memory_buffer[f"{prefix}proprioception"][
                            current_idx:end_idx
                        ]
                        hand_next = proprio_ep.copy()
                        if hand_next.shape[0] > 1:
                            hand_next[:-1] = proprio_ep[1:]
                        if (
                            hand_next.shape[1:]
                            != self.memory_buffer[f"{prefix}hand_action"].shape[1:]
                        ):
                            raise ValueError(
                                "action_from_next_state expects the joint "
                                "proprioception and hand_action to share the same "
                                f"feature dim, got proprioception "
                                f"{hand_next.shape[1:]} vs hand_action "
                                f"{self.memory_buffer[f'{prefix}hand_action'].shape[1:]}. "
                                "This mode only supports 22-D joint hand actions."
                            )
                        self.memory_buffer[f"{prefix}hand_action"][
                            current_idx:end_idx
                        ] = hand_next

                    # ``synthesize_next_state``: the SAME shift, but written to
                    # its own streams so the recorded command survives and both
                    # can be predicted as separate output blocks.
                    if self.synthesize_next_state:
                        pose_ep = self.memory_buffer[f"{prefix}pose"][
                            current_idx:end_idx
                        ]
                        pose_next = pose_ep.copy()
                        if pose_next.shape[0] > 1:
                            pose_next[:-1] = pose_ep[1:]
                        self.memory_buffer[f"{prefix}pose_next"][
                            current_idx:end_idx
                        ] = pose_next
                        proprio_ep = self.memory_buffer[f"{prefix}proprioception"][
                            current_idx:end_idx
                        ]
                        joint_next = proprio_ep.copy()
                        if joint_next.shape[0] > 1:
                            joint_next[:-1] = proprio_ep[1:]
                        self.memory_buffer[f"{prefix}joint_next"][
                            current_idx:end_idx
                        ] = joint_next

                    # Load fingertip_pose_wrist (T, 5, 6) STATE if needed.
                    if self.load_fingertip:
                        self.memory_buffer[f"{prefix}fingertip_pose_wrist"][
                            current_idx:end_idx
                        ] = self.load_low_dim_data(
                            root, osp.join(episode, f"{prefix}fingertip_pose_wrist")
                        )

                    # Load fingertip_action (T, 5, 6) ACTION (shift +1) if needed.
                    if self.load_fingertip_action:
                        self.memory_buffer[f"{prefix}fingertip_action"][
                            current_idx:end_idx
                        ] = self.load_low_dim_data(
                            root, osp.join(episode, f"{prefix}fingertip_action")
                        )

                    # Load FSR / tactile if enabled (from the configured field).
                    # The stream is stored under ``f"{prefix}{fsr_source_key}"``
                    # (e.g. "fsr" single-arm, "left_fsr" / "right_fsr" dual-arm)
                    # so stats.pickle, the batch dict and deploy can all refer to
                    # the tactile stream by its real name.
                    if self.enable_fsr:
                        fsr_data = self.load_low_dim_data(
                            root, osp.join(episode, f"{prefix}{self.fsr_source_key}")
                        )
                        # Per-episode baseline calibration (before binarize/norm)
                        # so the stream is the true delta above the per-episode
                        # resting level.  Two modes, gated by fsr_baseline_correct:
                        #   * mean-of-first-N (fsr_baseline_embodiments, e.g. human)
                        #   * first-frame / N=1 (fsr_baseline_first_frame_embodiments,
                        #     e.g. robot) -- first-frame takes precedence.
                        _emb = self.embodiments[path_idx]
                        _use_first = _emb in self.fsr_baseline_first_frame_embodiments
                        _use_mean = _emb in self.fsr_baseline_embodiments
                        if self.fsr_baseline_correct and (_use_first or _use_mean):
                            n = 1 if _use_first else self.fsr_baseline_frames
                            n = min(n, fsr_data.shape[0])
                            if n > 0:
                                baseline = fsr_data[:n].mean(axis=0, keepdims=True)
                                # Clamp at 0: the raw tactile is non-negative, so
                                # "0 = no contact".  Whenever the leading frames
                                # are not truly at rest (the hand already resting
                                # on the object) the baseline OVER-subtracts and
                                # the stream would go negative, which (a) invents
                                # a "less than no contact" region that has no
                                # physical meaning and (b) breaks the alignment
                                # with the robot stream, which is never corrected
                                # and therefore stays >= 0.  Clipping keeps both
                                # embodiments on the same "0 = no contact" scale.
                                fsr_data = np.maximum(fsr_data - baseline, 0.0)
                        if self.fsr_binarize:
                            # Binarize the tactile reading using the cutoff values.
                            fsr_data = np.where(
                                fsr_data >= self.fsr_binary_cutoff, 1.0, 0.0
                            )
                        fsr_data = fsr_data.astype(np.float32)
                        self.memory_buffer[f"{prefix}{self.fsr_source_key}"][
                            current_idx:end_idx
                        ] = fsr_data

                # Load camera data
                for camera_id in self.load_camera_ids:
                    cam_name = f"camera_{camera_id}"
                    self.memory_buffer[cam_name][current_idx:end_idx] = (
                        self.load_visual_data(root, osp.join(episode, cam_name, "rgb"))
                    )

                # Apply ring overlay (only for embodiments that lack the
                # camera ring physically — e.g. exoskeleton, manus).
                # We paint after camera_resize_shape has already been
                # applied so the overlay just needs to match the resized
                # image size.
                if apply_ring:
                    for camera_id in self.load_camera_ids:
                        cam_name = f"camera_{camera_id}"
                        self._ring_overlay.apply_batch(
                            self.memory_buffer[cam_name][current_idx:end_idx],
                            in_place=True,
                        )

                # Update episode end indices
                if load_episode_num == 0:
                    self.eps_end = [episode_length]
                else:
                    self.eps_end.append(self.eps_end[-1] + episode_length)
                self.episode_data_dir_idx.append(path_idx)
                self.episode_names.append(str(episode))

                current_idx = end_idx
                load_episode_num += 1

                if (
                    self.max_episode is not None
                    and load_episode_num >= self.max_episode
                ):
                    break

            if self.max_episode is not None and load_episode_num >= self.max_episode:
                break

        # Convert eps_end to numpy array for faster indexing
        self.eps_end = np.array(self.eps_end)
        self.episode_data_dir_idx = np.array(
            self.episode_data_dir_idx, dtype=np.int64
        )
        self.episode_names = np.array(self.episode_names, dtype=object)
        # Per-episode canonical embodiment string (same length/order as
        # ``eps_end``), derived from the data_dir each episode came from.
        # Consumed by per-embodiment normalization stats.
        self.episode_embodiment = np.array(
            [self.embodiments[i] for i in self.episode_data_dir_idx],
            dtype=object,
        )

        print(f"Loaded {load_episode_num} episodes with {total_frames} frames")

    def _preallocate_arrays(self, total_frames):
        """Pre-allocate arrays with the exact size needed to avoid memory fragmentation"""
        # Get sample shapes by loading first frame of first episode
        root = zarr.open(self.data_path[0], mode="r")
        episode = list(root.group_keys())[0]

        if self.action_from_next_state and self.skip_proprioception:
            raise ValueError(
                "action_from_next_state=True needs the proprioception "
                "stream (the joint action is proprioception[t+1]); set "
                "skip_proprioception=False (UVTADataset forces this)."
            )
        if self.synthesize_next_state and self.skip_proprioception:
            raise ValueError(
                "synthesize_next_state=True needs the proprioception stream "
                "(joint_next is proprioception[t+1]); set "
                "skip_proprioception=False (UVTADataset forces this)."
            )

        # Auto-detect the optional pose_action (real wrist ACTION) stream from
        # the FIRST arm (all arms are symmetric).  When ``action_from_next_state``
        # is on we SYNTHESIZE pose_action from the shifted ``pose`` state instead.
        if self.action_from_next_state:
            self.has_pose_action = True
        else:
            self.has_pose_action = (
                f"{self.arm_prefixes[0]}pose_action" in root[episode]
            )

        # Per-arm low-dim streams.  Single-arm data (``arm_prefixes == [""]``)
        # keeps the bare keys (``hand_action`` / ``pose`` / ...); dual-arm data
        # allocates ``left_*`` / ``right_*`` variants.
        for prefix in self.arm_prefixes:
            hand_action_shape = root[
                osp.join(episode, f"{prefix}hand_action")
            ].shape[1:]
            pose_shape = root[osp.join(episode, f"{prefix}pose")].shape[1:]

            self.memory_buffer[f"{prefix}hand_action"] = np.zeros(
                (total_frames, *hand_action_shape), dtype=np.float32
            )
            self.memory_buffer[f"{prefix}pose"] = np.zeros(
                (total_frames, *pose_shape), dtype=np.float32
            )

            # pose_action: synthesized (pose shape) under action_from_next_state,
            # else read from the data when present.
            if self.action_from_next_state:
                self.memory_buffer[f"{prefix}pose_action"] = np.zeros(
                    (total_frames, *pose_shape), dtype=np.float32
                )
            elif self.has_pose_action:
                pose_action_shape = root[
                    osp.join(episode, f"{prefix}pose_action")
                ].shape[1:]
                self.memory_buffer[f"{prefix}pose_action"] = np.zeros(
                    (total_frames, *pose_action_shape), dtype=np.float32
                )

            # Pre-allocate array for proprioception if needed
            if not self.skip_proprioception:
                proprio_shape = root[
                    osp.join(episode, f"{prefix}proprioception")
                ].shape[1:]
                self.memory_buffer[f"{prefix}proprioception"] = np.zeros(
                    (total_frames, *proprio_shape), dtype=np.float32
                )

            # The next-state streams mirror the shapes they are shifted from.
            if self.synthesize_next_state:
                self.memory_buffer[f"{prefix}pose_next"] = np.zeros(
                    (total_frames, *pose_shape), dtype=np.float32
                )
                self.memory_buffer[f"{prefix}joint_next"] = np.zeros(
                    (total_frames, *proprio_shape), dtype=np.float32
                )

            # Pre-allocate array for fingertip_pose_wrist (T, 5, 6) STATE.
            if self.load_fingertip:
                ft_shape = root[
                    osp.join(episode, f"{prefix}fingertip_pose_wrist")
                ].shape[1:]
                assert ft_shape == (5, 6), (
                    f"unexpected fingertip_pose_wrist shape: (T, *{ft_shape})"
                )
                self.memory_buffer[f"{prefix}fingertip_pose_wrist"] = np.zeros(
                    (total_frames, *ft_shape), dtype=np.float32
                )

            # Pre-allocate array for fingertip_action (T, 5, 6) ACTION (shift +1).
            if self.load_fingertip_action:
                fa_shape = root[
                    osp.join(episode, f"{prefix}fingertip_action")
                ].shape[1:]
                assert fa_shape == (5, 6), (
                    f"unexpected fingertip_action shape: (T, *{fa_shape})"
                )
                self.memory_buffer[f"{prefix}fingertip_action"] = np.zeros(
                    (total_frames, *fa_shape), dtype=np.float32
                )

            # Pre-allocate array for FSR / tactile if enabled.  Stored under
            # ``f"{prefix}{fsr_source_key}"``, see load_data_to_memory.
            if self.enable_fsr:
                fsr_shape = root[
                    osp.join(episode, f"{prefix}{self.fsr_source_key}")
                ].shape[1:]
                self.memory_buffer[f"{prefix}{self.fsr_source_key}"] = np.zeros(
                    (total_frames, *fsr_shape), dtype=np.float32
                )

        # Pre-allocate arrays for camera data
        for camera_id in self.load_camera_ids:
            cam_name = f"camera_{camera_id}"
            rgb_path = osp.join(episode, cam_name, "rgb")

            # Get the shape of a single image
            img_shape = root[rgb_path].shape[1:]

            # Adjust shape if resize is required
            if self.camera_resize_shape:
                img_shape = (*self.camera_resize_shape, img_shape[2])

            # Pre-allocate camera array
            self.memory_buffer[cam_name] = np.zeros(
                (total_frames, *img_shape), dtype=np.uint8
            )

        # Initialize eps_end as a list
        self.eps_end = []
