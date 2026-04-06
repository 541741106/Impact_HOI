from typing import Any, Dict, List, Optional
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QAbstractItemView,
    QFileDialog,
    QMessageBox,
    QComboBox,
    QToolButton,
    QSpinBox,
    QSlider,
    QCheckBox,
    QShortcut,
    QMenu,
    QGroupBox,
    QFormLayout,
    QLineEdit,
    QSplitter,
    QListView,
    QSizePolicy,
    QInputDialog,
    QFrame,
    QDialog,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QDialogButtonBox,
    QRadioButton,
    QButtonGroup,
    QProgressDialog,
    QApplication,
    QTabWidget,
    QScrollArea,
)
from ui.mixins import FrameControlMixin
from PyQt5.QtCore import Qt, QSize, QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import QStyle
from PyQt5.QtGui import QKeySequence, QColor
import copy
import json
import hashlib
from ui.video_player import VideoPlayer
from ui.label_panel import LabelPanel
from core.models import LabelDef
from core.hoi_query_controller import (
    apply_field_suggestion,
    build_query_candidates,
    clear_field_suggestion,
    clear_field_value,
    ensure_hand_annotation_state,
    get_field_state,
    hydrate_existing_field_state,
    set_field_confirmation,
    set_field_suggestion,
    _safe_int,
    _UNSET,
)
from core.structured_event_graph import build_hoi_event_graph, save_event_graph_sidecar
from utils.constants import PRESET_COLORS, color_from_key
from utils.shortcut_settings import (
    load_shortcut_bindings,
    default_shortcut_bindings,
    shortcut_value,
    set_shortcut_key,
    load_ui_preferences,
    save_ui_preferences,
    save_logging_policy,
)
from utils.op_logger import OperationLogger
from ui.hoi_timeline import HOITimeline
from ui.widgets import ToggleSwitch, ClickToggleList
import os
import cv2
import re
import yaml
import time
from datetime import datetime
from core.videomae_v2_logic import VideoMAEHandler



class EditableTitleGroupBox(QGroupBox):
    """Group box whose title can be renamed by double-clicking the title band."""

    titleEdited = pyqtSignal(str)

    def __init__(self, title="", parent=None):
        super().__init__(title, parent)
        self._title_editor = QLineEdit(self)
        self._title_editor.hide()
        self._title_editor.editingFinished.connect(self._finish_title_edit)
        self.setToolTip("Double-click the title to rename it.")

    def _title_edit_geometry(self):
        fm = self.fontMetrics()
        left = 12
        top = 2
        height = max(24, fm.height() + 10)
        width = max(
            180,
            min(self.width() - 24, fm.horizontalAdvance(self.title() or "Title") + 36),
        )
        return left, top, width, height

    def _begin_title_edit(self):
        left, top, width, height = self._title_edit_geometry()
        self._title_editor.setGeometry(left, top, width, height)
        self._title_editor.setText(self.title())
        self._title_editor.show()
        self._title_editor.raise_()
        self._title_editor.setFocus()
        self._title_editor.selectAll()

    def _finish_title_edit(self):
        if not self._title_editor.isVisible():
            return
        new_title = self._title_editor.text().strip() or self.title()
        self._title_editor.hide()
        if new_title != self.title():
            self.setTitle(new_title)
            self.titleEdited.emit(new_title)

    def mouseDoubleClickEvent(self, event):
        left, top, width, height = self._title_edit_geometry()
        if left <= event.pos().x() <= left + width and top <= event.pos().y() <= top + height:
            self._begin_title_edit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._title_editor.isVisible():
            left, top, width, height = self._title_edit_geometry()
            self._title_editor.setGeometry(left, top, width, height)


class ActionLabelSelector(QDialog):
    def __init__(self, candidates, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Action Label")
        self.layout = QVBoxLayout(self)
        self.selected_label = None
        
        self.layout.addWidget(QLabel("Multiple high-confidence actions detected. Please select one:"))
        
        self.group = QButtonGroup(self)
        for i, cand in enumerate(candidates):
            rb = QRadioButton(f"{cand['label']} ({cand['score']:.2%})")
            if i == 0:
                rb.setChecked(True)
            self.group.addButton(rb, i)
            self.layout.addWidget(rb)
        
        self.btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.btns.accepted.connect(self.accept)
        self.btns.rejected.connect(self.reject)
        self.layout.addWidget(self.btns)
        
        self.candidates = candidates

    def get_selected_label(self):
        idx = self.group.checkedId()
        if idx >= 0:
            return self.candidates[idx]['label']
        return self.candidates[0]['label']

class YoloTrainDialog(QDialog):
    def __init__(self, parent=None, default_epochs=10):
        super().__init__(parent)
        self.setWindowTitle("YOLO Incremental Training Config")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout(self)
        form = QFormLayout()
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(default_epochs)
        form.addRow("Epochs:", self.epochs_spin)
        
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(8)
        form.addRow("Batch Size:", self.batch_spin)

        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(320, 1280)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(640)
        form.addRow("Image Size:", self.imgsz_spin)

        self.lr_edit = QLineEdit("0.01")
        form.addRow("Learning Rate (Fine-tuning):", self.lr_edit)
        
        self.train_split_spin = QSpinBox()
        self.train_split_spin.setRange(50, 95)
        self.train_split_spin.setValue(80)
        self.train_split_spin.setSuffix("% Train")
        form.addRow("Train/Val Split:", self.train_split_spin)

        self.output_dir_edit = QLineEdit(os.path.join(os.getcwd(), "incremental_train"))
        btn_out = QPushButton("...")
        btn_out.setFixedWidth(30)
        btn_out.clicked.connect(self._choose_out)
        h_out = QHBoxLayout()
        h_out.addWidget(self.output_dir_edit)
        h_out.addWidget(btn_out)
        form.addRow("Dataset Output Dir:", h_out)
        
        layout.addLayout(form)
        
        self.btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.btn_box.accepted.connect(self.accept)
        self.btn_box.rejected.connect(self.reject)
        layout.addWidget(self.btn_box)

    def _choose_out(self):
        d = QFileDialog.getExistingDirectory(self, "Choose Output Directory")
        if d:
            self.output_dir_edit.setText(d)

    def get_config(self):
        return {
            "epochs": self.epochs_spin.value(),
            "batch": self.batch_spin.value(),
            "imgsz": self.imgsz_spin.value(),
            "lr0": float(self.lr_edit.text() or 0.01),
            "train_split": self.train_split_spin.value() / 100.0,
            "output_dir": self.output_dir_edit.text()
        }


class YoloTrainWorker(QThread):
    finished = pyqtSignal(bool, str)
    progress = pyqtSignal(str)

    def __init__(self, weights_path, data_yaml, config):
        super().__init__()
        self.weights_path = weights_path
        self.data_yaml = data_yaml
        self.config = config

    def run(self):
        try:
            from ultralytics import YOLO
            import torch
            
            self.progress.emit("Loading model...")
            model = YOLO(self.weights_path)
            
            self.progress.emit(f"Starting training for {self.config['epochs']} epochs...")
            # Note: project and name combined determine save dir
            results = model.train(
                data=self.data_yaml,
                epochs=self.config['epochs'],
                imgsz=self.config['imgsz'],
                batch=self.config['batch'],
                lr0=self.config['lr0'],
                device='cuda' if torch.cuda.is_available() else 'cpu',
                project=self.config['output_dir'],
                name='train_run',
                workers=0, # Avoid multiprocessing issues on Windows
                exist_ok=True
            )
            self.finished.emit(True, f"Training complete. Results saved to {results.save_dir}")
        except Exception as e:
            self.finished.emit(False, str(e))


class HOIWindow(FrameControlMixin, QWidget):
    """
    HandOI/HOI detection annotator:
    - Single video.
    - Load YOLO bbox text (per-line: frame class xc yc w h [conf]; normalized if <=1).
    - Overlay bboxes on video; show current-frame boxes in a list.
    - Define verbs via LabelPanel; select two boxes + a verb to create/delete relations.
    """
    extra_label_config = {
        "title": "Hand Anomaly Label",
        "default_label": "Normal",
        "labels": ["Normal"],
        "rules": {}
    }

    def __init__(
        self,
        parent=None,
        on_close=None,
        on_switch_task=None,
        tasks: List[str] = None,
        logger: OperationLogger = None,
    ):
        super().__init__(parent)
        # --- Customizable Extra Label Module (Default: Hand Anomaly) ---
        self.extra_label_config = {
            "title": "Hand Anomaly Label",
            "default_label": "Normal",
            "labels": [
                "Visibility / Occlusion",
                "Handling Anomaly",
                "Tool Anomaly",
                "Part Mismatch",
                "Spatial Anomaly",
                "Temporal Anomaly",
                "Quality / Outcome-visible Defect",
                "Completion / Finish Missing",
                "Detach / Fall-out",
                "Recovery / Rework",
                "Normal",
            ],
            "rules": {
                "Visibility / Occlusion": {
                    "allow_missing_bbox": True,
                    "allow_missing_verb": True,
                }
            },
        }

        self.setFocusPolicy(Qt.StrongFocus)
        self.setWindowTitle("HandOI / HOI Detection")
        self.resize(1280, 840)
        self._on_close = on_close
        self._on_switch_task = on_switch_task
        self.op_logger = logger or OperationLogger(False)
        self._ui_preferences = load_ui_preferences(default_ui_scale=0.85)
        self._ui_scale = float(self._ui_preferences.get("ui_scale", 0.85) or 0.85)
        self._shortcut_bindings = load_shortcut_bindings()
        self._shortcut_defaults = default_shortcut_bindings()

        self.player = VideoPlayer()
        self.player.on_frame_advanced = self._on_frame_advanced
        self.player.on_playback_state_changed = self._on_player_playback_state_changed

        # --- Customizable Actor Module (Default: Left/Right Hand) ---
        self.actors_config = [
            {"id": "Left_hand", "label": "Left Hand", "short": "L"},
            {"id": "Right_hand", "label": "Right Hand", "short": "R"},
        ]

        # data
        self.bboxes: Dict[int, List[dict]] = {}  # frame -> list of boxes
        self.box_id_counter = 1
        self.verbs: List[LabelDef] = []
        self.current_verb_idx = -1
        self._action_panel_sync = False
        self._pending_field_sources: Dict[str, str] = {}
        self.events: List[dict] = []  # Store all events (internal event structure)
        self.event_id_counter = 0  # Event ID counter
        self._reset_event_draft()  # Initialize event draft structure
        self.class_map: Dict[int, str] = {}
        self.labels_dir: str = ""
        self.start_offset = 0
        self.end_frame = None  # optional clamp
        self.raw_boxes: List[dict] = []  # keep raw frame indices for re-mapping
        self.current_hands = {actor["id"]: None for actor in self.actors_config}
        self.selected_hand_label = None  # Current active actor ID
        self.selected_event_id = None
        self._required_loaded = False
        self.video_path = ""
        # YOLO current-frame detection
        self.yolo_model = None
        self.yolo_weights_path = ""
        self.yolo_conf = 0.25
        self.yolo_iou = 0.45
        self.yolo_existing_policy = None  # "append" or "replace"
        self.mp_hands = None
        self.mp_hands_error = None
        self.mp_hands_max = 2
        self.mp_hands_conf = 0.5
        self.mp_hands_track_conf = 0.5
        self.mp_hands_swap = False

        # VideoMAE V2 action detection
        self.videomae_handler = VideoMAEHandler()
        self.videomae_weights_path = ""
        self.videomae_verb_list_path = ""
        self._videomae_loaded_key = ""
        self._videomae_action_cache: Dict[str, List[dict]] = {}
        self._videomae_event_signatures: Dict[int, str] = {}
        self._videomae_auto_event_id: Optional[int] = None
        self._videomae_auto_force_refresh = False
        self._videomae_auto_timer = QTimer(self)
        self._videomae_auto_timer.setSingleShot(True)
        self._videomae_auto_timer.setInterval(400)
        self._videomae_auto_timer.timeout.connect(
            self._run_pending_action_label_refresh
        )
        self._hoi_undo_stack = []
        self._hoi_redo_stack = []
        self._undo_block = False
        self._undo_limit = 50
        self.validation_enabled = False
        self.validation_session_active = False
        self.validation_started_at = None
        self.validation_modified = False
        self.validation_change_count = 0
        self.validator_name = ""
        self.validation_summary_enabled = True
        self.validation_op_logger = OperationLogger(self.op_logger.enabled)
        self._validation_highlights = {}
        self._hoi_saved_signature = None
        self._experiment_mode = "full_assist"
        self._next_best_query: Optional[Dict[str, Any]] = None
        self._next_best_query_id = ""
        self._next_best_query_presented_at = 0.0

        # controls (professional top chrome)
        self.toolbar_frame = QFrame(self)
        self.toolbar_frame.setObjectName("toolbarFrame")
        toolbar_layout = QVBoxLayout(self.toolbar_frame)
        toolbar_layout.setContentsMargins(8, 6, 8, 6)
        toolbar_layout.setSpacing(4)

        session_row = QHBoxLayout()
        session_row.setContentsMargins(0, 0, 0, 0)
        session_row.setSpacing(6)
        transport_row = QHBoxLayout()
        transport_row.setContentsMargins(0, 0, 0, 0)
        transport_row.setSpacing(6)

        lbl_task = QLabel("Task")
        lbl_task.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        session_row.addWidget(lbl_task)
        self.combo_task = QComboBox()
        items = tasks or ["Action Segmentation", "HandOI / HOI Detection"]
        self.combo_task.addItems(items)
        self.combo_task.setCurrentText("HandOI / HOI Detection")
        self.combo_task.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.combo_task.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        self.combo_task.currentTextChanged.connect(self._on_task_combo_changed)
        session_row.addWidget(self.combo_task)
        if len(items) <= 1:
            lbl_task.hide()
            self.combo_task.hide()

        lbl_experiment_mode = QLabel("Mode")
        lbl_experiment_mode.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        session_row.addWidget(lbl_experiment_mode)
        self.combo_experiment_mode = QComboBox()
        self.combo_experiment_mode.addItem("Manual", "manual")
        self.combo_experiment_mode.addItem("Assist", "assist")
        self.combo_experiment_mode.addItem("Full Assist", "full_assist")
        self.combo_experiment_mode.setCurrentIndex(2)
        self.combo_experiment_mode.setMinimumWidth(120)
        self.combo_experiment_mode.setToolTip(
            "Manual: no imported/detected assist. Assist: detection and box assist only. Full Assist: detection plus VideoMAE action-label ranking."
        )
        self.combo_experiment_mode.currentIndexChanged.connect(
            self._on_experiment_mode_changed
        )
        session_row.addWidget(self.combo_experiment_mode)
        session_row.addStretch(1)

        self.lbl_validation = QLabel("Validate")
        self.lbl_validation.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        session_row.addWidget(self.lbl_validation)
        self.btn_validation = ToggleSwitch(self)
        self.btn_validation.setToolTip("Toggle validation on/off")
        self.btn_validation.toggled.connect(self._on_validation_toggled)
        session_row.addWidget(self.btn_validation)

        self.file_menu = QMenu(self)
        self.file_menu.addAction("Load Video...", self._load_video)
        self.file_menu.addSeparator()
        import_menu = self.file_menu.addMenu("Import")
        import_menu.addAction("Instrument List...", self._import_instruments)
        import_menu.addAction("Target List...", self._import_targets)
        import_menu.addAction("Verb List...", self._load_verbs_txt)
        import_menu.addSeparator()
        import_menu.addAction("Class Map (data.yaml)...", self._load_yaml)
        self.act_import_yolo_boxes = import_menu.addAction(
            "YOLO Boxes...", self._load_bboxes
        )
        self.act_load_hands_xml = import_menu.addAction(
            "Hands XML...", self._load_hands_xml
        )
        import_menu.addSeparator()
        import_menu.addAction("HOI Annotations...", self._load_annotations_json)

        detect_menu = self.file_menu.addMenu("Detection")
        self.act_load_yolo_model = detect_menu.addAction(
            "Load YOLO Model...", self._load_yolo_model
        )
        self.act_detect_current_frame = detect_menu.addAction(
            "Detect Current Frame", self._detect_current_frame_combined
        )
        self.act_detect_selected_action = detect_menu.addAction(
            "Detect Selected Action", self._detect_selected_action
        )
        self.act_detect_all_actions = detect_menu.addAction(
            "Detect All Actions", self._detect_all_actions
        )
        self.act_swap_hands = detect_menu.addAction("Auto Swap Left / Right")
        self.act_swap_hands.setCheckable(True)
        self.act_swap_hands.toggled.connect(
            lambda on: setattr(self, "mp_hands_swap", on)
        )
        detect_menu.addSeparator()
        self.act_incremental_train_yolo = detect_menu.addAction(
            "YOLO Incremental Training...", self._incremental_train_yolo
        )

        assist_menu = self.file_menu.addMenu("Action Assist")
        self.act_load_videomae_model = assist_menu.addAction(
            "Load VideoMAE Model...", self._load_videomae_model
        )
        self.act_load_videomae_verb_list = assist_menu.addAction(
            "Load VideoMAE Verb List...", self._load_videomae_verb_list
        )
        assist_menu.addSeparator()
        self.act_review_selected_action_label = assist_menu.addAction(
            "Review Selected Action Label...", self._detect_selected_action_label
        )
        self.act_auto_apply_action_labels = assist_menu.addAction(
            "Auto-Apply Action Labels to All Events", self._detect_all_action_labels
        )

        save_menu = self.file_menu.addMenu("Save / Export")
        save_menu.addAction("Save HOI Annotations...", self._save_annotations_json)
        save_menu.addAction("Export Hands XML...", self._save_hands_xml)
        self.file_menu.addSeparator()
        self.file_menu.addAction("Settings...", self._open_settings_dialog)

        self.btn_file_menu = QToolButton()
        self.btn_file_menu.setText("\u22EF")
        self.btn_file_menu.setToolTip(
            "Project, import/export, detection, and model actions"
        )
        self.btn_file_menu.setPopupMode(QToolButton.InstantPopup)
        self.btn_file_menu.setMenu(self.file_menu)
        self.btn_file_menu.setFixedWidth(30)
        session_row.addWidget(self.btn_file_menu)

        self.btn_rew = QToolButton()
        self.btn_rew.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekBackward))
        self.btn_play = QToolButton()
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_play.setToolTip("Play")
        self.btn_stop = QToolButton()
        self.btn_stop.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.btn_ff = QToolButton()
        self.btn_ff.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekForward))
        for b in (
            self.btn_rew,
            self.btn_play,
            self.btn_stop,
            self.btn_ff,
        ):
            b.setToolButtonStyle(Qt.ToolButtonIconOnly)
            b.setIconSize(QSize(20, 20))
        self.spin_jump = QSpinBox()
        self.spin_jump.setMinimum(0)
        self.spin_jump.setMaximum(0)
        self.spin_jump.setKeyboardTracking(False)
        self.btn_jump = QToolButton()
        self.btn_jump.setText("\u2192")
        self.btn_jump.setToolTip("Jump to frame")

        self.btn_rew.clicked.connect(lambda: self._seek_relative(-10))
        self.btn_ff.clicked.connect(lambda: self._seek_relative(+10))
        self.btn_play.clicked.connect(self._toggle_play_pause)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_jump.clicked.connect(self._jump_to_spin)

        self.combo_verb = QComboBox()
        self.combo_verb.setMinimumWidth(96)
        self.combo_verb.setEditable(True)

        self.btn_detect = QToolButton(self)
        self.btn_detect.setText("Frame")
        self.btn_detect.setToolTip(
            "Detect objects + hands on current frame (Ctrl+Shift+D)"
        )
        self.btn_detect.clicked.connect(self._detect_current_frame_combined)
        self.btn_detect_action = QToolButton(self)
        self.btn_detect_action.setText("Action")
        self.btn_detect_action.setToolTip(
            "Detect assigned objects on the selected action's start, onset, and end frames."
        )
        self.btn_detect_action.clicked.connect(self._detect_selected_action)

        transport_row.addWidget(self.btn_rew)
        transport_row.addWidget(self.btn_play)
        transport_row.addWidget(self.btn_stop)
        transport_row.addWidget(self.btn_ff)
        transport_row.addSpacing(8)
        transport_row.addWidget(QLabel("Jump"))
        transport_row.addWidget(self.spin_jump)
        transport_row.addWidget(self.btn_jump)
        transport_row.addSpacing(8)
        transport_row.addWidget(QLabel("Start"))
        self.spin_start_offset = QSpinBox()
        self.spin_start_offset.setMinimum(0)
        self.spin_start_offset.valueChanged.connect(self._on_offset_changed)
        transport_row.addWidget(self.spin_start_offset)
        transport_row.addWidget(QLabel("End"))
        self.spin_end_frame = QSpinBox()
        self.spin_end_frame.setMinimum(0)
        self.spin_end_frame.setMaximum(10**9)
        self.spin_end_frame.valueChanged.connect(self._on_offset_changed)
        transport_row.addWidget(self.spin_end_frame)
        transport_row.addStretch(1)

        self.chk_edit_boxes = QCheckBox("\u270E Boxes")
        self.chk_edit_boxes.setToolTip("Enable direct box editing on the video canvas.")
        self.chk_edit_boxes.toggled.connect(self._on_edit_boxes_toggled)

        self.draw_mode_widget = QWidget(self)
        draw_mode_row = QHBoxLayout(self.draw_mode_widget)
        draw_mode_row.setContentsMargins(0, 0, 0, 0)
        draw_mode_row.setSpacing(6)
        lbl_draw_mode = QLabel("Draw")
        lbl_draw_mode.setObjectName("captionLabel")
        draw_mode_row.addWidget(lbl_draw_mode)
        self.rad_draw_none = QRadioButton("Manual")
        self.rad_draw_inst = QRadioButton("Inst")
        self.rad_draw_target = QRadioButton("Target")
        self.rad_draw_none.setChecked(True)
        self.rad_draw_none.setToolTip("Draw boxes and enter labels manually.")
        self.rad_draw_inst.setToolTip(
            "New boxes inherit the selected hand's instrument label."
        )
        self.rad_draw_target.setToolTip(
            "New boxes inherit the selected hand's target label."
        )
        draw_mode_row.addWidget(self.rad_draw_none)
        draw_mode_row.addWidget(self.rad_draw_inst)
        draw_mode_row.addWidget(self.rad_draw_target)
        for widget in (self.rad_draw_none, self.rad_draw_inst, self.rad_draw_target):
            widget.setEnabled(False)
        self.draw_mode_widget.hide()

        toolbar_layout.addLayout(session_row)
        toolbar_layout.addLayout(transport_row)

        # video row (main canvas above, inspector below)
        video_row = QHBoxLayout()
        video_row.setContentsMargins(0, 0, 0, 0)
        video_row.setSpacing(8)
        video_row.addWidget(self.player, 1)

        self.list_objects = QListWidget()
        self.list_objects.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list_objects.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_objects.customContextMenuRequested.connect(self._on_object_list_menu)
        self.list_objects.itemSelectionChanged.connect(self._on_object_selection)
        self.list_objects.setEnabled(False)
        self.list_objects.setAlternatingRowColors(True)
        self.list_objects.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)

        # --- Customizable Extra Label Module UI setup moved to top of __init__ ---
        self.group_anomaly = EditableTitleGroupBox(
            self.extra_label_config.get("title", "Hand Anomaly Label")
        )
        self.group_anomaly.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.group_anomaly.titleEdited.connect(self._on_anomaly_title_edited)
        self.anomaly_labels = list(self.extra_label_config.get("labels", []))
        self.anomaly_rules = {}
        self._init_anomaly_rules()
        self.anomaly_list = ClickToggleList()
        self.anomaly_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.anomaly_list.setEditTriggers(QAbstractItemView.EditKeyPressed)
        self.anomaly_list.setFlow(QListView.LeftToRight)
        self.anomaly_list.setWrapping(True)
        self.anomaly_list.setResizeMode(QListView.Adjust)
        self.anomaly_list.setSpacing(10)
        self.anomaly_list.setStyleSheet(
            "QListWidget::item { padding: 3px 10px; }"
            "QListWidget::indicator { width: 14px; height: 14px; }"
        )
        self.anomaly_list.itemChanged.connect(self._on_anomaly_item_changed)
        self.anomaly_list.bodyDoubleClicked.connect(self._on_anomaly_item_double_clicked)
        self.anomaly_list.setToolTip("Double-click a label to rename it.")
        self.anomaly_list.setWordWrap(True)
        self.anomaly_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        self._anomaly_block = False
        for name in self.anomaly_labels:
            self._add_anomaly_item(name, checked=(name.strip().lower() == self.extra_label_config.get("default_label", "Normal").strip().lower()))

        self.anomaly_edit = QLineEdit(self.group_anomaly)
        self.anomaly_edit.setPlaceholderText("New label")
        self.btn_anomaly_add = QPushButton("Add")
        self.btn_anomaly_remove = QPushButton("Remove")
        self.btn_anomaly_rules = QPushButton("Rules")
        self.btn_anomaly_add.clicked.connect(self._add_anomaly_label)
        self.btn_anomaly_remove.clicked.connect(self._remove_anomaly_label)
        self.btn_anomaly_rules.clicked.connect(self._edit_anomaly_rules)

        anomaly_layout = QVBoxLayout()
        anomaly_layout.addWidget(self.anomaly_list)
        row = QHBoxLayout()
        row.addWidget(self.anomaly_edit, 1)
        row.addWidget(self.btn_anomaly_add)
        row.addWidget(self.btn_anomaly_remove)
        row.addWidget(self.btn_anomaly_rules)
        anomaly_layout.addLayout(row)
        self.group_anomaly.setLayout(anomaly_layout)

        hand_row = QHBoxLayout()
        hand_row.setContentsMargins(0, 0, 0, 0)
        hand_row.setSpacing(8)
        lbl_actors = QLabel("Actors")
        lbl_actors.setObjectName("captionLabel")
        hand_row.addWidget(lbl_actors)

        self.actor_layout = QHBoxLayout()
        hand_row.addLayout(self.actor_layout)

        self.actor_controls = {}
        self._rebuild_actor_checkboxes()

        self.btn_config_actors = QPushButton("Manage")
        self.btn_config_actors.setToolTip("Configure actors (Add/Remove/Rename)")
        self.btn_config_actors.clicked.connect(self._on_configure_actors)
        hand_row.addWidget(self.btn_config_actors)

        self.btn_swap_draft = QPushButton("\u21C4")
        self.btn_swap_draft.setToolTip(
            "Swap actor boxes on the current frame (first two actors)."
        )
        self.btn_swap_draft.setStyleSheet("color: #111;")
        self.btn_swap_draft.clicked.connect(self._swap_frame_hands)
        hand_row.addWidget(self.btn_swap_draft)
        hand_row.addStretch(1)

        self.label_panel = LabelPanel(
            self.verbs,
            on_add=self._on_verb_added,
            on_remove_idx=self._on_verb_removed,
            on_rename=self._on_verb_renamed,
            on_search_matches=None,
            on_select_idx=self._on_verb_selected,
            manage_storage=False,
        )

        for le in self.label_panel.findChildren(QLineEdit):
            ph = le.placeholderText().lower()
            if "search" in ph:
                le.setPlaceholderText("Search action library")
            elif "new" in ph:
                le.setPlaceholderText("Add action to library")
        self._init_verb_color_combo()
        self.label_panel.set_verb_only(True)
        self.label_panel.set_admin_visible(False)
        self.label_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)

        self.group_library = QGroupBox("Objects")
        self.group_library.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        lib_layout = QVBoxLayout()
        lib_layout.setContentsMargins(8, 8, 8, 8)
        lib_layout.setSpacing(6)

        object_tools_row = QHBoxLayout()
        object_tools_row.setContentsMargins(0, 0, 0, 0)
        object_tools_row.setSpacing(6)
        object_tools_row.addWidget(self.btn_detect)
        object_tools_row.addWidget(self.btn_detect_action)
        self.btn_object_tools = QToolButton()
        self.btn_object_tools.setText("⋯")
        self.btn_object_tools.setPopupMode(QToolButton.InstantPopup)
        self.btn_object_tools.setToolTip("More object tools")
        self.object_tools_menu = QMenu(self.btn_object_tools)
        self.object_tools_menu.addAction("Detect All", self._detect_all_actions)
        self.object_tools_menu.addSeparator()
        self.object_tools_menu.addAction(self.act_load_yolo_model)
        self.object_tools_menu.addAction(self.act_incremental_train_yolo)
        self.object_tools_menu.addSeparator()
        self.object_tools_menu.addAction(self.act_swap_hands)
        self.btn_object_tools.setMenu(self.object_tools_menu)
        object_tools_row.addWidget(self.btn_object_tools)
        object_tools_row.addStretch(1)
        lib_layout.addLayout(object_tools_row)
        edit_tools_row = QHBoxLayout()
        edit_tools_row.setContentsMargins(0, 0, 0, 0)
        edit_tools_row.setSpacing(6)
        edit_tools_row.addWidget(self.chk_edit_boxes)
        edit_tools_row.addStretch(1)
        lib_layout.addLayout(edit_tools_row)
        lib_layout.addWidget(self.draw_mode_widget)

        link_form = QFormLayout()
        self.combo_instrument = QComboBox()
        self.combo_target = QComboBox()
        self.combo_instrument.addItem("None", None)
        self.combo_target.addItem("None", None)

        self._enable_combo_search(
            self.combo_instrument, placeholder="Search instrument..."
        )
        self._enable_combo_search(self.combo_target, placeholder="Search target...")

        link_form.addRow("Instrument", self.combo_instrument)
        link_form.addRow("Target", self.combo_target)
        lib_layout.addLayout(link_form)

        lbl_current_objects = QLabel("Current Frame")
        lbl_current_objects.setObjectName("captionLabel")
        lib_layout.addWidget(lbl_current_objects)
        lib_layout.addWidget(self.list_objects, 1)
        self.group_library.setLayout(lib_layout)

        hand_widget = QWidget()
        hand_layout = QHBoxLayout(hand_widget)
        hand_layout.setContentsMargins(0, 0, 0, 0)
        hand_layout.addLayout(hand_row)

        self.group_action = QGroupBox("Action")
        self.group_action.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        action_layout = QVBoxLayout(self.group_action)
        action_layout.setContentsMargins(8, 8, 8, 8)
        action_layout.setSpacing(6)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.addWidget(QLabel("Final"))
        self.combo_verb.setPlaceholderText("Select or type action...")
        action_row.addWidget(self.combo_verb, 1)
        self.btn_suggest_action_label = QToolButton()
        self.btn_suggest_action_label.setText("\u21BB")
        self.btn_suggest_action_label.setToolTip(
            "Force a fresh VideoMAE ranking for the selected HOI event. Ranking also updates automatically when you select or retime an event."
        )
        self.btn_suggest_action_label.clicked.connect(self._refresh_selected_action_label)
        self.btn_action_tools = QToolButton()
        self.btn_action_tools.setText("\u22EF")
        self.btn_action_tools.setPopupMode(QToolButton.InstantPopup)
        self.btn_action_tools.setToolTip("Action tools")
        self.action_tools_menu = QMenu(self.btn_action_tools)
        self.act_toggle_verb_library_admin = self.action_tools_menu.addAction(
            "Manage Verb Library"
        )
        self.act_toggle_verb_library_admin.setCheckable(True)
        self.act_toggle_verb_library_admin.toggled.connect(
            self._set_verb_library_admin_mode
        )
        self.action_tools_menu.addSeparator()
        self.act_batch_apply_action_labels = self.action_tools_menu.addAction(
            "Auto-Apply Top-1 to All Events", self._detect_all_action_labels
        )
        self.btn_action_tools.setMenu(self.action_tools_menu)
        action_row.addWidget(self.btn_suggest_action_label)
        action_row.addWidget(self.btn_action_tools)
        action_layout.addLayout(action_row)
        action_layout.addWidget(self.label_panel, 1)

        self.label_panel.verb_list.itemClicked.connect(self._on_verb_library_item_clicked)
        self.label_panel.verb_list.setToolTip(
            "Click a verb to apply it to the current action."
        )

        self.event_status_card = QFrame(self)
        self.event_status_card.setObjectName("statusCard")
        event_status_layout = QVBoxLayout(self.event_status_card)
        event_status_layout.setContentsMargins(12, 12, 12, 12)
        event_status_layout.setSpacing(8)
        event_header = QHBoxLayout()
        event_header.setContentsMargins(0, 0, 0, 0)
        self.lbl_event_title = QLabel("No event selected")
        self.lbl_event_title.setObjectName("statusTitle")
        event_header.addWidget(self.lbl_event_title, 1)
        self.lbl_event_actor_chip = QLabel("Idle")
        self.lbl_event_actor_chip.setAlignment(Qt.AlignCenter)
        event_header.addWidget(self.lbl_event_actor_chip)
        self.lbl_event_health_chip = QLabel("Waiting")
        self.lbl_event_health_chip.setAlignment(Qt.AlignCenter)
        event_header.addWidget(self.lbl_event_health_chip)
        event_status_layout.addLayout(event_header)
        self.lbl_event_frames = QLabel("Frames: ?")
        self.lbl_event_frames.setObjectName("statusSubtle")
        event_status_layout.addWidget(self.lbl_event_frames)
        self.lbl_event_meta = QLabel("Verb: ?   Instrument: ?   Target: ?")
        self.lbl_event_meta.setWordWrap(True)
        event_status_layout.addWidget(self.lbl_event_meta)
        self.lbl_event_status = QLabel("No HandOI segment selected.")
        self.lbl_event_status.setWordWrap(True)
        self.lbl_event_status.setObjectName("statusSubtle")
        event_status_layout.addWidget(self.lbl_event_status)
        keyframe_row = QHBoxLayout()
        keyframe_row.setContentsMargins(0, 0, 0, 0)
        keyframe_row.setSpacing(6)
        self.btn_jump_start_chip = QPushButton("Start")
        self.btn_jump_start_chip.setProperty("compactChip", True)
        self.btn_jump_start_chip.clicked.connect(self._jump_to_selected_start)
        self.btn_jump_onset_chip = QPushButton("Onset")
        self.btn_jump_onset_chip.setProperty("compactChip", True)
        self.btn_jump_onset_chip.clicked.connect(self._jump_to_selected_onset)
        self.btn_jump_end_chip = QPushButton("End")
        self.btn_jump_end_chip.setProperty("compactChip", True)
        self.btn_jump_end_chip.clicked.connect(self._jump_to_selected_end)
        keyframe_row.addWidget(self.btn_jump_start_chip)
        keyframe_row.addWidget(self.btn_jump_onset_chip)
        keyframe_row.addWidget(self.btn_jump_end_chip)
        keyframe_row.addStretch(1)
        event_status_layout.addLayout(keyframe_row)

        self.event_tab = QWidget()
        event_layout = QVBoxLayout(self.event_tab)
        event_layout.setContentsMargins(0, 0, 0, 0)
        event_layout.setSpacing(8)
        event_layout.addWidget(self.event_status_card)
        event_layout.addWidget(hand_widget)
        event_layout.addWidget(self.group_action)
        event_layout.addStretch(1)

        self.objects_tab = QWidget()
        objects_layout = QVBoxLayout(self.objects_tab)
        objects_layout.setContentsMargins(0, 0, 0, 0)
        objects_layout.setSpacing(8)
        objects_layout.addWidget(self.group_library)
        objects_layout.addStretch(1)

        self.review_status_card = QFrame(self)
        self.review_status_card.setObjectName("statusCard")
        review_status_layout = QVBoxLayout(self.review_status_card)
        review_status_layout.setContentsMargins(12, 12, 12, 12)
        review_status_layout.setSpacing(8)
        review_header = QHBoxLayout()
        review_header.setContentsMargins(0, 0, 0, 0)
        self.lbl_review_title = QLabel("Review Summary")
        self.lbl_review_title.setObjectName("statusTitle")
        review_header.addWidget(self.lbl_review_title, 1)
        self.lbl_validation_chip = QLabel("Validation Off")
        self.lbl_validation_chip.setAlignment(Qt.AlignCenter)
        review_header.addWidget(self.lbl_validation_chip)
        self.lbl_incomplete_chip = QLabel("Incomplete n/a")
        self.lbl_incomplete_chip.setAlignment(Qt.AlignCenter)
        review_header.addWidget(self.lbl_incomplete_chip)
        review_status_layout.addLayout(review_header)
        self.lbl_review_status = QLabel("No review issues yet.")
        self.lbl_review_status.setObjectName("statusSubtle")
        self.lbl_review_status.setWordWrap(True)
        review_status_layout.addWidget(self.lbl_review_status)
        review_nav_row = QHBoxLayout()
        review_nav_row.setContentsMargins(0, 0, 0, 0)
        review_nav_row.setSpacing(6)
        self.btn_review_prev = QPushButton("Prev")
        self.btn_review_prev.clicked.connect(lambda: self._jump_incomplete(-1))
        self.btn_review_next = QPushButton("Next")
        self.btn_review_next.clicked.connect(lambda: self._jump_incomplete(+1))
        review_nav_row.addWidget(self.btn_review_prev)
        review_nav_row.addWidget(self.btn_review_next)
        review_nav_row.addStretch(1)
        review_status_layout.addLayout(review_nav_row)

        self.next_query_card = QFrame(self)
        self.next_query_card.setObjectName("statusCard")
        next_query_layout = QVBoxLayout(self.next_query_card)
        next_query_layout.setContentsMargins(12, 12, 12, 12)
        next_query_layout.setSpacing(8)
        next_query_header = QHBoxLayout()
        next_query_header.setContentsMargins(0, 0, 0, 0)
        self.lbl_next_query_title = QLabel("Next Best Query")
        self.lbl_next_query_title.setObjectName("statusTitle")
        next_query_header.addWidget(self.lbl_next_query_title, 1)
        self.lbl_next_query_surface_chip = QLabel("Idle")
        self.lbl_next_query_surface_chip.setAlignment(Qt.AlignCenter)
        next_query_header.addWidget(self.lbl_next_query_surface_chip)
        self.lbl_next_query_score_chip = QLabel("VOI --")
        self.lbl_next_query_score_chip.setAlignment(Qt.AlignCenter)
        next_query_header.addWidget(self.lbl_next_query_score_chip)
        next_query_layout.addLayout(next_query_header)
        self.lbl_next_query_summary = QLabel("No pending query yet.")
        self.lbl_next_query_summary.setWordWrap(True)
        next_query_layout.addWidget(self.lbl_next_query_summary)
        self.lbl_next_query_reason = QLabel("The controller will surface the most valuable next supervision step here.")
        self.lbl_next_query_reason.setObjectName("statusSubtle")
        self.lbl_next_query_reason.setWordWrap(True)
        next_query_layout.addWidget(self.lbl_next_query_reason)
        next_query_metrics_row = QHBoxLayout()
        next_query_metrics_row.setContentsMargins(0, 0, 0, 0)
        next_query_metrics_row.setSpacing(6)
        self.lbl_next_query_prop_chip = QLabel("Prop --")
        self.lbl_next_query_prop_chip.setAlignment(Qt.AlignCenter)
        next_query_metrics_row.addWidget(self.lbl_next_query_prop_chip)
        self.lbl_next_query_cost_chip = QLabel("Cost --")
        self.lbl_next_query_cost_chip.setAlignment(Qt.AlignCenter)
        next_query_metrics_row.addWidget(self.lbl_next_query_cost_chip)
        self.lbl_next_query_risk_chip = QLabel("Risk --")
        self.lbl_next_query_risk_chip.setAlignment(Qt.AlignCenter)
        next_query_metrics_row.addWidget(self.lbl_next_query_risk_chip)
        next_query_metrics_row.addStretch(1)
        next_query_layout.addLayout(next_query_metrics_row)
        self.lbl_next_query_evidence = QLabel("Sparse evidence: --")
        self.lbl_next_query_evidence.setObjectName("statusSubtle")
        self.lbl_next_query_evidence.setWordWrap(True)
        next_query_layout.addWidget(self.lbl_next_query_evidence)
        next_query_action_row = QHBoxLayout()
        next_query_action_row.setContentsMargins(0, 0, 0, 0)
        next_query_action_row.setSpacing(6)
        self.btn_next_query_focus = QPushButton("Focus")
        self.btn_next_query_focus.clicked.connect(self._focus_next_best_query)
        self.btn_next_query_apply = QPushButton("Apply")
        self.btn_next_query_apply.clicked.connect(self._apply_next_best_query)
        next_query_action_row.addWidget(self.btn_next_query_focus)
        next_query_action_row.addWidget(self.btn_next_query_apply)
        next_query_action_row.addStretch(1)
        next_query_layout.addLayout(next_query_action_row)

        self.review_tab = QWidget()
        review_layout = QVBoxLayout(self.review_tab)
        review_layout.setContentsMargins(0, 0, 0, 0)
        review_layout.setSpacing(8)
        review_layout.addWidget(self.review_status_card)
        review_layout.addWidget(self.next_query_card)
        review_layout.addWidget(self.group_anomaly)
        review_layout.addStretch(1)

        self.event_tab_scroll = self._make_inspector_scroll(self.event_tab)
        self.objects_tab_scroll = self._make_inspector_scroll(self.objects_tab)
        self.review_tab_scroll = self._make_inspector_scroll(self.review_tab)

        self.inspector_tabs = QTabWidget(self)
        self.inspector_tabs.setObjectName("inspectorTabs")
        self.inspector_tabs.setDocumentMode(True)
        self.inspector_tabs.addTab(self.event_tab_scroll, "Event")
        self.inspector_tabs.addTab(self.objects_tab_scroll, "Objects")
        self.inspector_tabs.addTab(self.review_tab_scroll, "Review")
        self.inspector_tabs.setMinimumWidth(240)
        self.inspector_tabs.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        top_block = QWidget(self)
        top_layout = QVBoxLayout(top_block)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(6)
        top_layout.addLayout(video_row, 1)
        top_layout.addWidget(self.toolbar_frame)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self._on_slider_changed)
        top_layout.addWidget(self.slider)
        self.hoi_timeline = HOITimeline(
            get_segments_for_hand=self._hoi_segments_for_hand,
            get_color_for_verb=self._hoi_color_for_verb,
            on_select=self._on_hoi_timeline_select,
            on_update=self._on_hoi_timeline_update,
            on_create=self._on_hoi_timeline_create,
            on_delete=self._on_hoi_timeline_delete,
            on_hover=self._on_hoi_timeline_hover,
            get_frame_count=lambda: int(self.player.frame_count or 0),
            get_fps=lambda: int(self.player.frame_rate or 30),
            get_title_for_hand=self._hoi_title_for_hand,
            actors_config=self.actors_config,
            parent=self,
        )
        self.lbl_incomplete = QLabel("Incomplete: n/a")
        self.lbl_incomplete.setToolTip("No HandOI completeness check available yet.")
        self.btn_incomplete_prev = QToolButton()
        self.btn_incomplete_prev.setText("<")
        self.btn_incomplete_prev.setToolTip("Previous incomplete segment (A)")
        self.btn_incomplete_prev.clicked.connect(lambda: self._jump_incomplete(-1))
        self.btn_incomplete_next = QToolButton()
        self.btn_incomplete_next.setText(">")
        self.btn_incomplete_next.setToolTip("Next incomplete segment (D)")
        self.btn_incomplete_next.clicked.connect(lambda: self._jump_incomplete(+1))
        self.btn_incomplete_prev.setEnabled(False)
        self.btn_incomplete_next.setEnabled(False)
        self._incomplete_issues = []
        self._incomplete_idx = -1

        timeline_container = QWidget()
        timeline_layout = QVBoxLayout(timeline_container)
        timeline_layout.setContentsMargins(0, 0, 0, 0)
        timeline_layout.setSpacing(4)
        timeline_layout.addWidget(self.hoi_timeline, 1)
        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.addStretch(1)
        footer.addWidget(self.lbl_incomplete)
        footer.addSpacing(6)
        footer.addWidget(self.btn_incomplete_prev)
        footer.addWidget(self.btn_incomplete_next)
        timeline_layout.addLayout(footer)
        timeline_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.hoi_bottom_split = QSplitter(Qt.Horizontal, self)
        self.hoi_bottom_split.setChildrenCollapsible(False)
        self.hoi_bottom_split.setHandleWidth(10)
        self.hoi_bottom_split.setOpaqueResize(False)
        self.hoi_bottom_split.addWidget(self.inspector_tabs)
        self.hoi_bottom_split.addWidget(timeline_container)
        self.hoi_bottom_split.setStretchFactor(0, 0)
        self.hoi_bottom_split.setStretchFactor(1, 1)
        self.hoi_bottom_split.setSizes([280, 980])
        self.hoi_bottom_split.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.root_split = QSplitter(Qt.Vertical, self)
        self.root_split.setChildrenCollapsible(False)
        self.root_split.setHandleWidth(10)
        self.root_split.setOpaqueResize(False)
        self.root_split.addWidget(top_block)
        self.root_split.addWidget(self.hoi_bottom_split)
        self.root_split.setStretchFactor(0, 5)
        self.root_split.setStretchFactor(1, 2)
        self.root_split.setSizes([700, 200])
        root.addWidget(self.root_split, 1)
        self.chk_filter_current = None

        # shortcuts (only within HOI window)
        self.sc_left = QShortcut(
            QKeySequence(Qt.Key_Left), self, activated=lambda: self._seek_relative(-1)
        )
        self.sc_left.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_right = QShortcut(
            QKeySequence(Qt.Key_Right), self, activated=lambda: self._seek_relative(+1)
        )
        self.sc_right.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_up = QShortcut(
            QKeySequence(Qt.Key_Up), self, activated=lambda: self._seek_seconds(-1)
        )
        self.sc_up.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_down = QShortcut(
            QKeySequence(Qt.Key_Down), self, activated=lambda: self._seek_seconds(+1)
        )
        self.sc_down.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_toggle_play = QShortcut(
            QKeySequence(Qt.Key_Space), self, activated=self._toggle_play_pause
        )
        self.sc_toggle_play.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_pause = QShortcut(QKeySequence("K"), self, activated=self._pause)
        self.sc_pause.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_detect = QShortcut(
            QKeySequence("Ctrl+Shift+D"),
            self,
            activated=self._detect_current_frame_combined,
        )
        self.sc_detect.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_jump_start = QShortcut(
            QKeySequence("Z"), self, activated=self._jump_to_selected_start
        )
        self.sc_jump_start.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_jump_onset = QShortcut(
            QKeySequence("X"), self, activated=self._jump_to_selected_onset
        )
        self.sc_jump_onset.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_jump_end = QShortcut(
            QKeySequence("C"), self, activated=self._jump_to_selected_end
        )
        self.sc_jump_end.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_inc_prev = QShortcut(
            QKeySequence("A"), self, activated=lambda: self._jump_incomplete(-1)
        )
        self.sc_inc_prev.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_inc_next = QShortcut(
            QKeySequence("D"), self, activated=lambda: self._jump_incomplete(+1)
        )
        self.sc_inc_next.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_undo = QShortcut(QKeySequence.Undo, self, activated=self._hoi_undo)
        self.sc_undo.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_redo = QShortcut(QKeySequence.Redo, self, activated=self._hoi_redo)
        self.sc_redo.setContext(Qt.WidgetWithChildrenShortcut)
        self.sc_open_settings = QShortcut(QKeySequence("Ctrl+,"), self, activated=self._open_settings_dialog)
        self.sc_open_settings.setContext(Qt.WidgetWithChildrenShortcut)
        self.apply_shortcut_settings(self._shortcut_bindings)

        # 3. Safely handle list_objects signals
        try:
            self.list_objects.itemSelectionChanged.disconnect()
        except (TypeError, RuntimeError):
            pass

        # 4. Connect new logic
        self.list_objects.itemSelectionChanged.connect(self._on_object_selection)
        # --- New: Global Object ID Registry ---
        # Structure: { "screwdriver_1": 0, "gear_1": 1, ... }
        self.global_object_map: Dict[str, int] = {}
        self.object_id_counter = 0

        self.entity_library = {}
        self.id_to_category = {}

        self.combo_verb.currentTextChanged.connect(self._on_hoi_meta_changed)
        self.combo_instrument.currentIndexChanged.connect(self._on_hoi_meta_changed)
        self.combo_target.currentIndexChanged.connect(self._on_hoi_meta_changed)
        self._mark_hoi_saved()

        self._update_verb_combo()
        self._set_verb_library_admin_mode(False)
        self._update_draw_mode_visibility()
        self._set_ui_scale(getattr(self, "_ui_scale", 0.85), persist=False)
        self._apply_experiment_mode_ui()
        self._set_validation_ui_state(False)
        self._update_incomplete_indicator()
        self._update_status_label()
        self._update_next_best_query_panel()
        self._update_play_pause_button()

    def _make_inspector_scroll(self, content: QWidget) -> QScrollArea:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        scroll.setWidget(content)
        return scroll

    def _normalized_ui_scale(self, value: Optional[float] = None) -> float:
        try:
            scale = float(self._ui_scale if value is None else value)
        except Exception:
            scale = 0.85
        if scale > 10.0:
            scale = scale / 100.0
        return max(0.80, min(1.25, scale))

    def _scaled_ui_px(self, base: float, min_px: int = 0) -> int:
        try:
            px = int(round(float(base) * self._normalized_ui_scale()))
        except Exception:
            px = int(round(float(base)))
        return max(int(min_px), px)

    def _reset_workspace_layout(self) -> None:
        total_w = max(1000, int(self.width() or 0) - 20)
        inspector_w = min(320, max(250, int(total_w * 0.23)))
        total_h = max(720, int(self.height() or 0) - 40)
        bottom_h = max(220, int(total_h * 0.24))
        top_h = max(460, total_h - bottom_h)
        try:
            self.hoi_bottom_split.setSizes([inspector_w, max(720, total_w - inspector_w)])
        except Exception:
            pass
        try:
            self.root_split.setSizes([top_h, bottom_h])
        except Exception:
            pass

    def _apply_logging_settings(self, oplog_enabled: bool, validation_summary_enabled: bool) -> None:
        handled = False
        parent = self.parentWidget()
        if parent is not None and hasattr(parent, "_set_logging_policy"):
            try:
                parent._set_logging_policy(bool(oplog_enabled), bool(validation_summary_enabled))
                handled = True
            except Exception:
                handled = False
        if not handled:
            self.set_logging_policy(bool(oplog_enabled), bool(validation_summary_enabled))
            ok_save, path_or_err = save_logging_policy({
                "ops_csv_enabled": bool(oplog_enabled),
                "validation_summary_enabled": bool(validation_summary_enabled),
            })
            if not ok_save:
                print(f"[LOG][ERROR] Failed to save logging policy: {path_or_err}")

    def _set_ui_scale(self, scale: float, persist: bool = False) -> None:
        self._ui_scale = self._normalized_ui_scale(scale)
        self._apply_professional_ui_style()
        self._apply_micro_interaction_icons()
        try:
            self.combo_experiment_mode.setMinimumWidth(self._scaled_ui_px(120, 104))
        except Exception:
            pass
        try:
            self.btn_file_menu.setFixedWidth(self._scaled_ui_px(30, 24))
        except Exception:
            pass
        try:
            self.hoi_bottom_split.setHandleWidth(self._scaled_ui_px(10, 8))
            self.root_split.setHandleWidth(self._scaled_ui_px(10, 8))
        except Exception:
            pass
        try:
            if getattr(self, "hoi_timeline", None):
                self.hoi_timeline.updateGeometry()
                self.hoi_timeline.refresh()
        except Exception:
            pass
        self.updateGeometry()
        self.update()
        if persist:
            ok_save, path_or_err = save_ui_preferences({"ui_scale": self._ui_scale})
            if not ok_save:
                print(f"[UI][ERROR] Failed to save UI preferences: {path_or_err}")
            try:
                self._log("hoi_ui_scale_update", ui_scale=self._ui_scale)
            except Exception:
                pass

    def _open_settings_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Settings")
        dialog.resize(420, 260)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        grp_appearance = QGroupBox("Appearance", dialog)
        form_appearance = QFormLayout(grp_appearance)
        form_appearance.setContentsMargins(12, 12, 12, 12)
        combo_scale = QComboBox(grp_appearance)
        options = [
            (0.80, "80%"),
            (0.85, "85%"),
            (0.90, "90%"),
            (1.00, "100%"),
            (1.10, "110%"),
            (1.20, "120%"),
        ]
        current_scale = self._normalized_ui_scale()
        current_idx = 0
        best_diff = 999.0
        for idx, (value, label) in enumerate(options):
            combo_scale.addItem(label, value)
            diff = abs(value - current_scale)
            if diff < best_diff:
                best_diff = diff
                current_idx = idx
        combo_scale.setCurrentIndex(current_idx)
        form_appearance.addRow("UI Scale", combo_scale)
        lbl_scale_hint = QLabel("Scales fonts, button density, and toolbar icons.", grp_appearance)
        lbl_scale_hint.setObjectName("statusSubtle")
        lbl_scale_hint.setWordWrap(True)
        form_appearance.addRow("", lbl_scale_hint)
        btn_reset_layout = QPushButton("Reset Layout", grp_appearance)
        btn_reset_layout.clicked.connect(self._reset_workspace_layout)
        form_appearance.addRow("Panels", btn_reset_layout)
        layout.addWidget(grp_appearance)

        grp_logging = QGroupBox("Evaluation Logging", dialog)
        logging_layout = QVBoxLayout(grp_logging)
        logging_layout.setContentsMargins(12, 12, 12, 12)
        chk_ops = QCheckBox("Write operations CSV", grp_logging)
        chk_ops.setChecked(bool(getattr(self.op_logger, "enabled", False)))
        chk_validation = QCheckBox("Write validation summary", grp_logging)
        chk_validation.setChecked(bool(getattr(self, "validation_summary_enabled", True)))
        logging_layout.addWidget(chk_ops)
        logging_layout.addWidget(chk_validation)
        layout.addWidget(grp_logging)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() != QDialog.Accepted:
            return

        self._set_ui_scale(float(combo_scale.currentData() or self._ui_scale), persist=True)
        self._apply_logging_settings(chk_ops.isChecked(), chk_validation.isChecked())

    def _set_verb_library_admin_mode(self, on: bool) -> None:
        on = bool(on)
        if hasattr(self, "label_panel") and self.label_panel:
            self.label_panel.set_admin_visible(on)
        act = getattr(self, "act_toggle_verb_library_admin", None)
        if act is not None and act.isChecked() != on:
            act.blockSignals(True)
            act.setChecked(on)
            act.blockSignals(False)

    def _set_status_chip(self, widget: Optional[QLabel], text: str, tone: str = "neutral") -> None:
        if widget is None:
            return
        palette = {
            "neutral": ("#EEF2FF", "#3730A3"),
            "ok": ("#ECFDF3", "#027A48"),
            "warn": ("#FFFAEB", "#B54708"),
            "danger": ("#FEF3F2", "#B42318"),
        }
        bg, fg = palette.get(tone, palette["neutral"])
        widget.setText(text)
        widget.setStyleSheet(
            f"background: {bg}; color: {fg}; border: 1px solid {bg}; border-radius: 999px; padding: 3px 10px; font-weight: 600;"
        )

    def _set_status_card_tone(self, widget: Optional[QFrame], tone: str = "neutral") -> None:
        if widget is None:
            return
        palette = {
            "neutral": ("#F8FAFC", "#E4E7EC"),
            "active": ("#EFF6FF", "#93C5FD"),
            "ok": ("#F0FDF4", "#86EFAC"),
            "warn": ("#FFFBEB", "#FCD34D"),
            "danger": ("#FEF2F2", "#FCA5A5"),
        }
        bg, border = palette.get(tone, palette["neutral"])
        widget.setStyleSheet(
            f"background: {bg}; border: 1px solid {border}; border-radius: 10px;"
        )

    def _focus_inspector_tab(self, name: str) -> None:
        tabs = getattr(self, "inspector_tabs", None)
        if tabs is None:
            return
        key = str(name or "").strip().lower()
        mapping = {"event": 0, "objects": 1, "review": 2}
        idx = mapping.get(key)
        if idx is not None and 0 <= idx < tabs.count():
            tabs.setCurrentIndex(idx)

    def _update_inspector_tab_labels(self) -> None:
        tabs = getattr(self, "inspector_tabs", None)
        if tabs is None or tabs.count() < 3:
            return
        event_text = "Event"
        if self.selected_event_id is not None:
            event_text += " *"
        object_list = getattr(self, "list_objects", None)
        object_count = int(object_list.count()) if object_list is not None else 0
        objects_text = f"Objects ({object_count})" if object_count > 0 else "Objects"
        issue_count = len(getattr(self, "_incomplete_issues", []) or [])
        review_text = f"Review ({issue_count})" if issue_count > 0 else "Review"
        tabs.setTabText(0, event_text)
        tabs.setTabText(1, objects_text)
        tabs.setTabText(2, review_text)

    def _apply_micro_interaction_icons(self) -> None:
        style = self.style()
        icon_px = self._scaled_ui_px(18, 14)
        transport_w = self._scaled_ui_px(32, 26)
        chrome_w = self._scaled_ui_px(30, 24)
        try:
            self.btn_file_menu.setIcon(style.standardIcon(QStyle.SP_DialogOpenButton))
            self.btn_file_menu.setText("")
            self.btn_file_menu.setToolButtonStyle(Qt.ToolButtonIconOnly)
            self.btn_file_menu.setIconSize(QSize(icon_px, icon_px))
            self.btn_file_menu.setFixedWidth(chrome_w)
            self.btn_file_menu.setAutoRaise(True)
        except Exception:
            pass
        try:
            for button in (self.btn_rew, self.btn_play, self.btn_stop, self.btn_ff):
                button.setIconSize(QSize(icon_px, icon_px))
                button.setFixedWidth(transport_w)
                button.setAutoRaise(True)
            self.btn_jump.setFixedWidth(self._scaled_ui_px(28, 24))
        except Exception:
            pass
        try:
            self.btn_suggest_action_label.setIcon(
                style.standardIcon(QStyle.SP_BrowserReload)
            )
            self.btn_suggest_action_label.setText("")
            self.btn_suggest_action_label.setToolButtonStyle(Qt.ToolButtonIconOnly)
            self.btn_suggest_action_label.setIconSize(QSize(icon_px, icon_px))
            self.btn_suggest_action_label.setAutoRaise(True)
        except Exception:
            pass
        try:
            self.btn_object_tools.setText("?")
            self.btn_object_tools.setToolButtonStyle(Qt.ToolButtonTextOnly)
            self.btn_object_tools.setAutoRaise(True)
            self.btn_action_tools.setText("...")
            self.btn_action_tools.setToolButtonStyle(Qt.ToolButtonTextOnly)
            self.btn_action_tools.setAutoRaise(True)
        except Exception:
            pass

    def _query_surface_tone(self, surface: str) -> str:
        key = str(surface or "").strip().lower()
        if key == "event":
            return "active"
        if key == "objects":
            return "warn"
        if key == "review":
            return "danger"
        return "neutral"

    def _query_surface_chip_tone(self, surface: str) -> str:
        key = str(surface or "").strip().lower()
        if key == "event":
            return "neutral"
        if key == "objects":
            return "warn"
        if key == "review":
            return "danger"
        return "neutral"

    def _build_hoi_query_rows(self) -> List[dict]:
        rows: List[dict] = []
        graph = build_hoi_event_graph(
            self.events,
            video_path=self.video_path,
            annotation_path="",
            actors_config=self.actors_config,
        )
        consistency_by_key: Dict[tuple, List[dict]] = {}
        for item in list(graph.get("consistency_flags", []) or []):
            if not isinstance(item, dict):
                continue
            key = (item.get("event_id"), item.get("actor_id") or item.get("hand"))
            consistency_by_key.setdefault(key, []).append(dict(item))

        for event in self.events:
            event_id = event.get("event_id")
            event_candidates = self._normalize_videomae_candidates(
                event.get("videomae_top5")
            )
            for actor in self.actors_config:
                hand_key = actor["id"]
                hand_data = event.get("hoi_data", {}).get(hand_key, {}) or {}
                self._hydrate_hand_annotation_state(
                    hand_data, default_source="loaded_annotation"
                )
                state = copy.deepcopy(hand_data.get("_field_state", {}) or {})
                suggestions = copy.deepcopy(
                    hand_data.get("_field_suggestions", {}) or {}
                )
                start = hand_data.get("interaction_start")
                onset = hand_data.get("functional_contact_onset")
                end = hand_data.get("interaction_end")
                verb = hand_data.get("verb")
                inst = hand_data.get("instrument_object_id")
                target = hand_data.get("target_object_id")
                has_info = any(
                    (
                        start is not None,
                        onset is not None,
                        end is not None,
                        bool(str(verb or "").strip()),
                        inst is not None,
                        target is not None,
                        bool(suggestions),
                    )
                )
                if not has_info:
                    continue

                labels = self._parse_anomaly_labels(hand_data.get("anomaly_label"))
                allow_missing_bbox = self._anomaly_rule_allows(
                    labels, "allow_missing_bbox"
                )
                sparse_evidence = self._sparse_evidence_summary(hand_data)
                bbox_errors = (
                    []
                    if allow_missing_bbox
                    else list(self._validate_integrity(hand_data) or [])
                )
                rows.append(
                    {
                        "event_id": event_id,
                        "hand": hand_key,
                        "interaction_start": start,
                        "functional_contact_onset": onset,
                        "interaction_end": end,
                        "verb": verb,
                        "instrument_object_id": inst,
                        "target_object_id": target,
                        "field_state": state,
                        "field_suggestions": suggestions,
                        "bbox_errors": bbox_errors,
                        "sparse_evidence_state": copy.deepcopy(
                            sparse_evidence.get("state", {}) or {}
                        ),
                        "sparse_evidence_summary": {
                            "expected": int(sparse_evidence.get("expected", 0) or 0),
                            "confirmed": int(
                                sparse_evidence.get("confirmed", 0) or 0
                            ),
                            "missing": int(sparse_evidence.get("missing", 0) or 0),
                            "blocked": int(sparse_evidence.get("blocked", 0) or 0),
                        },
                        "allow_missing_bbox": bool(allow_missing_bbox),
                        "videomae_candidates": list(event_candidates or []),
                        "consistency_flags": list(
                            consistency_by_key.get((event_id, hand_key), []) or []
                        ),
                    }
                )
        return rows

    def _update_next_best_query_panel(self) -> None:
        if not hasattr(self, "lbl_next_query_summary"):
            return
        queries = build_query_candidates(
            self._build_hoi_query_rows(),
            selected_event_id=self.selected_event_id,
            selected_hand=self.selected_hand_label,
        )
        best = dict(queries[0]) if queries else None
        self._next_best_query = best
        if not best:
            self._next_best_query_id = ""
            self._next_best_query_presented_at = 0.0
            self.lbl_next_query_title.setText("Next Best Query")
            self.lbl_next_query_summary.setText("No pending high-value query.")
            self.lbl_next_query_reason.setText(
                "All current HOI fields are either confirmed or intentionally left unresolved."
            )
            self.lbl_next_query_evidence.setText("Sparse evidence: all active keyframe slots are currently grounded.")
            self._set_status_chip(
                getattr(self, "lbl_next_query_surface_chip", None), "Idle", "neutral"
            )
            self._set_status_chip(
                getattr(self, "lbl_next_query_score_chip", None), "VOI --", "neutral"
            )
            self._set_status_chip(
                getattr(self, "lbl_next_query_prop_chip", None), "Prop --", "neutral"
            )
            self._set_status_chip(
                getattr(self, "lbl_next_query_cost_chip", None), "Cost --", "neutral"
            )
            self._set_status_chip(
                getattr(self, "lbl_next_query_risk_chip", None), "Risk --", "neutral"
            )
            self._set_status_card_tone(getattr(self, "next_query_card", None), "neutral")
            self.btn_next_query_focus.setEnabled(False)
            self.btn_next_query_apply.setEnabled(False)
            self.btn_next_query_apply.setText("Apply")
            return

        surface = str(best.get("surface") or "review").strip().title()
        event_id = best.get("event_id")
        hand = self._get_actor_short_label(best.get("hand"))
        query_id = str(best.get("query_id") or "").strip()
        if query_id and query_id != self._next_best_query_id:
            self._next_best_query_id = query_id
            self._next_best_query_presented_at = time.time()
            self._log(
                "hoi_query_present",
                query_id=query_id,
                query_type=best.get("query_type"),
                event_id=event_id,
                hand=best.get("hand"),
                surface=best.get("surface"),
                field=best.get("field_name"),
                target_frame=best.get("target_frame"),
                target_slot=best.get("target_slot"),
                voi_score=best.get("voi_score"),
                propagation_gain=best.get("propagation_gain"),
                human_cost_est=best.get("human_cost_est"),
                overwrite_risk=best.get("overwrite_risk"),
            )
        summary = str(best.get("summary") or "").strip()
        reason = str(best.get("reason") or "").strip()
        voi = float(best.get("voi_score", 0.0) or 0.0)
        propagation = float(best.get("propagation_gain", 0.0) or 0.0)
        cost = float(best.get("human_cost_est", 0.0) or 0.0)
        risk = float(best.get("overwrite_risk", 0.0) or 0.0)
        evidence_summary = dict(best.get("sparse_evidence_summary") or {})
        evidence_expected = int(evidence_summary.get("expected", 0) or 0)
        evidence_confirmed = int(evidence_summary.get("confirmed", 0) or 0)
        evidence_missing = int(evidence_summary.get("missing", 0) or 0)
        location = f"Event {event_id} {hand}" if event_id is not None else hand
        self.lbl_next_query_title.setText(f"Next Best Query  {location}".strip())
        self.lbl_next_query_summary.setText(summary or "Review the next supervision step.")
        self.lbl_next_query_reason.setText(reason or "Controller-selected high-value supervision step.")
        if evidence_expected > 0:
            self.lbl_next_query_evidence.setText(
                f"Sparse evidence: {evidence_confirmed}/{evidence_expected} grounded, {evidence_missing} still missing."
            )
        else:
            self.lbl_next_query_evidence.setText(
                "Sparse evidence: waiting for object links and keyframes before grounding can begin."
            )
        self._set_status_chip(
            getattr(self, "lbl_next_query_surface_chip", None),
            surface,
            self._query_surface_chip_tone(surface),
        )
        self._set_status_chip(
            getattr(self, "lbl_next_query_score_chip", None),
            f"VOI {voi:.2f}",
            "neutral",
        )
        self._set_status_chip(
            getattr(self, "lbl_next_query_prop_chip", None),
            f"Prop {propagation:.2f}",
            "ok" if propagation >= 0.85 else "neutral",
        )
        self._set_status_chip(
            getattr(self, "lbl_next_query_cost_chip", None),
            f"Cost {cost:.2f}",
            "warn" if cost >= 0.5 else "neutral",
        )
        self._set_status_chip(
            getattr(self, "lbl_next_query_risk_chip", None),
            f"Risk {risk:.2f}",
            "danger" if risk >= 0.3 else "neutral",
        )
        self._set_status_card_tone(
            getattr(self, "next_query_card", None),
            self._query_surface_tone(surface),
        )
        self.btn_next_query_focus.setEnabled(True)
        can_apply = bool(best.get("safe_apply"))
        self.btn_next_query_apply.setEnabled(can_apply)
        if best.get("apply_mode") == "confirm_current":
            self.btn_next_query_apply.setText("Confirm Current")
        else:
            self.btn_next_query_apply.setText("Apply Suggestion")

    def _focus_query_candidate(self, query: Optional[dict]) -> None:
        if not query:
            return
        event_id = query.get("event_id")
        hand = query.get("hand")
        if event_id is not None and hand:
            self._set_selected_event(int(event_id), str(hand))
        surface = str(query.get("surface") or "").strip().lower()
        if surface:
            self._focus_inspector_tab(surface)
        frame = query.get("target_frame")
        if frame is not None and self.player.cap:
            try:
                target = int(frame)
                self.player.seek(target)
                self._refresh_boxes_for_frame(target)
                self._set_frame_controls(target)
            except Exception:
                pass
        latency_ms = None
        query_id = str(query.get("query_id") or "").strip()
        if query_id and query_id == getattr(self, "_next_best_query_id", ""):
            presented_at = float(getattr(self, "_next_best_query_presented_at", 0.0) or 0.0)
            if presented_at > 0:
                latency_ms = int(max(0.0, (time.time() - presented_at) * 1000.0))
        self._log(
            "hoi_query_focus",
            query_id=query_id,
            query_type=query.get("query_type"),
            event_id=event_id,
            hand=hand,
            surface=surface,
            field=query.get("field_name"),
            target_frame=query.get("target_frame"),
            target_slot=query.get("target_slot"),
            voi_score=query.get("voi_score"),
            propagation_gain=query.get("propagation_gain"),
            human_cost_est=query.get("human_cost_est"),
            overwrite_risk=query.get("overwrite_risk"),
            query_latency_ms=latency_ms,
        )

    def _focus_next_best_query(self) -> None:
        self._focus_query_candidate(getattr(self, "_next_best_query", None))

    def _apply_query_suggestion(self, query: Optional[dict]) -> bool:
        if not query or not bool(query.get("safe_apply")):
            return False
        event_id = query.get("event_id")
        hand_key = query.get("hand")
        field_name = str(query.get("field_name") or "").strip()
        apply_mode = str(query.get("apply_mode") or "").strip()
        if event_id is None or not hand_key or not field_name:
            return False
        event = self._find_event_by_id(int(event_id))
        if not event:
            return False
        hand_data = event.get("hoi_data", {}).get(hand_key, {}) or {}
        self._ensure_hand_annotation_state(hand_data)
        draft = None
        if self.selected_event_id == int(event_id) and hand_key in self.event_draft:
            draft = self.event_draft.get(hand_key, {})
            self._ensure_hand_annotation_state(draft)

        applied = False
        if apply_mode == "confirm_current":
            state = get_field_state(hand_data, field_name)
            if state.get("status") != "confirmed":
                self._set_hand_field_state(
                    hand_data,
                    field_name,
                    source="query_confirm",
                    status="confirmed",
                )
                if isinstance(draft, dict):
                    self._set_hand_field_state(
                        draft,
                        field_name,
                        source="query_confirm",
                        status="confirmed",
                    )
                applied = True
        elif apply_mode == "apply_suggestion":
            suggested_value = query.get("suggested_value")
            suggested_source = (
                str(query.get("suggested_source") or "").strip() or "query_controller"
            )
            reason = str(query.get("reason") or "").strip()
            confidence = query.get("suggested_confidence")
            if get_field_state(hand_data, field_name).get("status") != "confirmed":
                self._suggest_hand_field(
                    hand_data,
                    field_name,
                    suggested_value,
                    source=suggested_source,
                    confidence=confidence,
                    reason=reason,
                    safe_to_apply=True,
                )
                applied = apply_field_suggestion(
                    hand_data,
                    field_name,
                    source=suggested_source,
                    as_status="suggested",
                )
                if applied and isinstance(draft, dict):
                    self._suggest_hand_field(
                        draft,
                        field_name,
                        suggested_value,
                        source=suggested_source,
                        confidence=confidence,
                        reason=reason,
                        safe_to_apply=True,
                    )
                    apply_field_suggestion(
                        draft,
                        field_name,
                        source=suggested_source,
                        as_status="suggested",
                    )

        if not applied:
            return False

        latency_ms = None
        query_id = str(query.get("query_id") or "").strip()
        if query_id and query_id == getattr(self, "_next_best_query_id", ""):
            presented_at = float(getattr(self, "_next_best_query_presented_at", 0.0) or 0.0)
            if presented_at > 0:
                latency_ms = int(max(0.0, (time.time() - presented_at) * 1000.0))
        self._sync_event_frames(event)
        if field_name == "verb":
            if self.selected_event_id == int(event_id) and self.selected_hand_label == hand_key:
                self._load_hand_draft_to_ui(hand_key)
            self._update_action_top5_display(int(event_id))
        else:
            if self.selected_event_id == int(event_id) and self.selected_hand_label == hand_key:
                self._update_status_label()
        self._refresh_events()
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.refresh()
        self._log(
            "hoi_query_apply",
            query_id=query_id,
            query_type=query.get("query_type"),
            event_id=event_id,
            hand=hand_key,
            field=field_name,
            apply_mode=apply_mode,
            target_frame=query.get("target_frame"),
            target_slot=query.get("target_slot"),
            voi_score=query.get("voi_score"),
            propagation_gain=query.get("propagation_gain"),
            human_cost_est=query.get("human_cost_est"),
            overwrite_risk=query.get("overwrite_risk"),
            resolve_kind="safe_apply",
            query_latency_ms=latency_ms,
        )
        return True

    def _apply_next_best_query(self) -> None:
        if not self._apply_query_suggestion(getattr(self, "_next_best_query", None)):
            QMessageBox.information(
                self,
                "Next Best Query",
                "This query requires explicit user confirmation rather than a safe local completion.",
            )

    def _update_draw_mode_visibility(self) -> None:
        draw_widget = getattr(self, "draw_mode_widget", None)
        if draw_widget is not None:
            draw_widget.setVisible(bool(getattr(self, "chk_edit_boxes", None) and self.chk_edit_boxes.isChecked()))

    def _preferred_base_font_pt(self) -> float:
        app = QApplication.instance()
        base_font = app.font() if app is not None else self.font()
        try:
            size = float(base_font.pointSizeF())
        except Exception:
            size = float(base_font.pointSize() or 0)
        if size <= 0:
            size = 8.0
        scaled = size * self._normalized_ui_scale()
        return max(6.8, min(8.4, scaled))

    def _apply_professional_ui_style(self) -> None:
        body_pt = self._preferred_base_font_pt()
        caption_pt = max(6.3, body_pt - 0.30)
        title_pt = body_pt
        scale = self._normalized_ui_scale()
        group_margin_top = self._scaled_ui_px(12, 10)
        group_pad_top = self._scaled_ui_px(4, 3)
        tab_pad_v = self._scaled_ui_px(3, 2)
        tab_pad_h = self._scaled_ui_px(8, 6)
        tab_min_w = self._scaled_ui_px(52, 48)
        control_h = self._scaled_ui_px(22, 20)
        control_radius = self._scaled_ui_px(7, 6)
        control_pad_v = self._scaled_ui_px(1.5, 1)
        control_pad_h = self._scaled_ui_px(5, 4)
        compact_h = self._scaled_ui_px(19, 17)
        compact_pad_h = self._scaled_ui_px(5, 4)
        list_item_h = self._scaled_ui_px(22, 20)
        slider_handle = self._scaled_ui_px(14, 12)
        slider_margin = self._scaled_ui_px(5, 4)
        font = self.font()
        try:
            current_size = float(font.pointSizeF())
        except Exception:
            current_size = float(font.pointSize() or 0)
        if abs(current_size - body_pt) > 0.05:
            font.setPointSizeF(body_pt)
            self.setFont(font)
        self.setStyleSheet(
            f"""
            QFrame#toolbarFrame {{
                background: #F8FAFC;
                border: 1px solid #E4E7EC;
                border-radius: 10px;
            }}
            QWidget, QFrame, QLabel, QGroupBox, QPushButton, QToolButton, QComboBox, QLineEdit, QSpinBox, QListWidget, QTabWidget {{
                font-family: "Segoe UI", "Microsoft YaHei UI", "Tahoma", "Arial";
            }}
            QGroupBox {{
                background: #FFFFFF;
                border: 1px solid #D9DEE7;
                border-radius: 10px;
                margin-top: {group_margin_top}px;
                padding-top: {group_pad_top}px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
                color: #344054;
                font-weight: 600;
            }}
            QLabel#captionLabel {{
                color: #667085;
                font-size: {caption_pt:.1f}pt;
                font-weight: 600;
                text-transform: uppercase;
            }}
            QTabWidget::pane {{
                border: 1px solid #D9DEE7;
                background: #FFFFFF;
                border-radius: 10px;
                top: -1px;
            }}
            QTabBar::tab {{
                background: #F2F4F7;
                border: 1px solid #D0D5DD;
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: {tab_pad_v}px {tab_pad_h}px;
                margin-right: 4px;
                color: #475467;
                min-width: {tab_min_w}px;
            }}
            QTabBar::tab:selected {{
                background: #FFFFFF;
                color: #101828;
                font-weight: 600;
            }}
            QSplitter::handle {{
                background: #E4E7EC;
                border-radius: 4px;
            }}
            QSplitter::handle:hover {{
                background: #CBD5E1;
            }}
            QSplitter::handle:horizontal {{
                width: 10px;
                margin: 0 1px;
            }}
            QSplitter::handle:vertical {{
                height: 10px;
                margin: 1px 0;
            }}
            QPushButton, QToolButton, QComboBox, QLineEdit, QSpinBox {{
                min-height: {control_h}px;
                border: 1px solid #D0D5DD;
                border-radius: {control_radius}px;
                background: #FFFFFF;
                padding: {control_pad_v}px {control_pad_h}px;
            }}
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            QFrame#statusCard {{
                background: #F8FAFC;
                border: 1px solid #E4E7EC;
                border-radius: 10px;
            }}
            QLabel#statusTitle {{
                color: #101828;
                font-size: {title_pt:.1f}pt;
                font-weight: 700;
            }}
            QLabel#statusSubtle {{
                color: #667085;
            }}
            QPushButton[compactChip="true"] {{
                min-height: {compact_h}px;
                padding: {control_pad_v}px {compact_pad_h}px;
            }}
            QPushButton:hover, QToolButton:hover {{
                background: #F9FAFB;
            }}
            QPushButton:disabled, QToolButton:disabled, QComboBox:disabled, QLineEdit:disabled, QSpinBox:disabled {{
                color: #98A2B3;
                background: #F2F4F7;
            }}
            QToolButton::menu-indicator {{
                image: none;
                width: 0px;
            }}
            QListWidget {{
                border: 1px solid #D0D5DD;
                border-radius: 8px;
                background: #FFFFFF;
                padding: 4px;
            }}
            QListWidget::item {{
                min-height: {control_h}px;
            }}
            QListWidget::item:selected {{
                background: #E0EAFF;
                color: #0F172A;
                border-radius: 6px;
            }}
            QSlider::groove:horizontal {{
                height: 6px;
                background: #E4E7EC;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                width: {slider_handle}px;
                margin: -{slider_margin}px 0;
                background: #2563EB;
                border-radius: {control_radius}px;
            }}
            """
        )

    def _set_shortcut_key(
        self, shortcut: Optional[QShortcut], sid: str, default_key: str
    ) -> None:
        set_shortcut_key(
            shortcut,
            shortcut_value(
                self._shortcut_bindings,
                self._shortcut_defaults,
                sid,
                default_key,
            ),
            default_key,
        )

    def apply_shortcut_settings(
        self, bindings: Optional[Dict[str, str]] = None
    ) -> None:
        self._shortcut_bindings = (
            load_shortcut_bindings() if bindings is None else dict(bindings)
        )
        self._shortcut_defaults = default_shortcut_bindings()
        self._set_shortcut_key(getattr(self, "sc_left", None), "hoi.step_prev", "Left")
        self._set_shortcut_key(
            getattr(self, "sc_right", None), "hoi.step_next", "Right"
        )
        self._set_shortcut_key(
            getattr(self, "sc_up", None), "hoi.seek_prev_second", "Up"
        )
        self._set_shortcut_key(
            getattr(self, "sc_down", None), "hoi.seek_next_second", "Down"
        )
        self._set_shortcut_key(
            getattr(self, "sc_toggle_play", None), "hoi.play_pause", "Space"
        )
        self._set_shortcut_key(getattr(self, "sc_pause", None), "hoi.pause", "K")
        self._set_shortcut_key(
            getattr(self, "sc_detect", None), "hoi.detect", "Ctrl+Shift+D"
        )
        self._set_shortcut_key(getattr(self, "sc_undo", None), "hoi.undo", "Ctrl+Z")
        self._set_shortcut_key(getattr(self, "sc_redo", None), "hoi.redo", "Ctrl+Y")

    # ---------- data helpers ----------
    def _enable_combo_search(self, combo: QComboBox, placeholder: str = "Search..."):
        combo.setEditable(True)
        combo.setInsertPolicy(QComboBox.NoInsert)
        combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        if combo.lineEdit() is not None:
            combo.lineEdit().setPlaceholderText(placeholder)
        completer = combo.completer()
        if completer is not None:
            completer.setCaseSensitivity(Qt.CaseInsensitive)
            try:
                completer.setFilterMode(Qt.MatchContains)
            except Exception:
                pass

    def _init_verb_color_combo(self):
        combo = self.label_panel.combo
        if combo.findText("Auto") == -1:
            combo.insertItem(0, "Auto")
        combo.setCurrentText("Auto")
        combo.currentTextChanged.connect(self._on_verb_color_changed)

    def _on_verb_color_changed(self, color_key: str):
        if color_key == "Custom":
            return
        selected = self.label_panel.current_label_name()
        if not selected:
            return
        if self.label_panel.edit.text().strip():
            return
        idx = self.label_panel.index_of_label(selected)
        if idx < 0:
            return
        self.verbs[idx].color_name = color_key
        self.label_panel.refresh()
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.refresh()

    def _update_verb_combo(self):
        """[Restore] Populate the verb combobox from self.verbs list."""
        current = self.combo_verb.currentText()
        self.combo_verb.blockSignals(True)
        self.combo_verb.clear()

        sorted_verbs = sorted([v.name for v in self.verbs])
        self.combo_verb.addItems(sorted_verbs)

        idx = self.combo_verb.findText(current)
        if idx >= 0:
            self.combo_verb.setCurrentIndex(idx)
        elif current:
            self.combo_verb.setCurrentText(current)

        self.combo_verb.blockSignals(False)
        self._sync_action_panel_selection(self.combo_verb.currentText())

    def _hoi_color_for_verb(self, verb: str) -> Optional[QColor]:
        for v in self.verbs:
            if v.name == verb:
                color_name = v.color_name or ""
                if color_name.lower() == "auto":
                    return None
                return color_from_key(color_name)
        return None

    def _find_event_by_id(self, event_id: int) -> Optional[dict]:
        for ev in self.events:
            if ev.get("event_id") == event_id:
                return ev
        return None

    def _compute_event_frames(self, event: dict) -> tuple:
        times = []
        for actor in self.actors_config:
            hand = actor["id"]
            h = event.get("hoi_data", {}).get(hand, {})
            for key in (
                "interaction_start",
                "functional_contact_onset",
                "interaction_end",
            ):
                val = h.get(key)
                if val is not None:
                    try:
                        times.append(int(val))
                    except Exception:
                        continue
        if not times:
            return None, None
        return min(times), max(times)

    def _sync_event_frames(self, event: dict):
        ws, we = self._compute_event_frames(event)
        event["frames"] = [ws, we]

    def _apply_draft_to_selected_event(self):
        if self.selected_event_id is None:
            return
        ev = self._find_event_by_id(self.selected_event_id)
        if not ev:
            return
        ev["hoi_data"] = {aid: dict(data) for aid, data in self.event_draft.items()}
        self._sync_event_frames(ev)

    def _apply_selected_event(self):
        if self.selected_event_id is None:
            return
        self._save_ui_to_hand_draft(self.selected_hand_label)
        self._apply_draft_to_selected_event()
        self._refresh_events()
        self._update_hoi_titles()
        self.hoi_timeline.refresh()

    def _set_selected_event(self, event_id: int, hand_key: Optional[str] = None):
        if self.selected_event_id == event_id and hand_key == self.selected_hand_label:
            return
        if self.selected_event_id is not None:
            self._save_ui_to_hand_draft(self.selected_hand_label)
            self._apply_draft_to_selected_event()

        ev = self._find_event_by_id(event_id)
        if not ev:
            return
        self._sync_event_frames(ev)
        self.event_id_counter = max(self.event_id_counter, event_id + 1)
        self.event_draft = {
            actor["id"]: dict(ev.get("hoi_data", {}).get(actor["id"], {}))
            for actor in self.actors_config
        }

        actor_ids = [a["id"] for a in self.actors_config]
        if hand_key not in actor_ids:
            hand_key = actor_ids[0]
            for aid in actor_ids:
                if (
                    ev.get("hoi_data", {})
                    .get(aid, {})
                    .get("interaction_start")
                    is not None
                ):
                    hand_key = aid
                    break
        self.selected_hand_label = hand_key
        self.selected_event_id = event_id

        for aid, chk in self.actor_controls.items():
            chk.blockSignals(True)
            chk.setChecked(aid == hand_key)
            chk.blockSignals(False)

        self._load_hand_draft_to_ui(hand_key)
        self._update_status_label()
        self.list_objects.setEnabled(True)

        if hasattr(self, "hoi_timeline") and self.hoi_timeline:
            self.hoi_timeline.set_selected(event_id, hand_key)
            self.hoi_timeline.refresh()

        self._update_action_top5_display(event_id)
        self._queue_action_label_refresh(event_id, delay_ms=250)
        self._update_overlay(self.player.current_frame)
        self._update_hoi_titles()

    def _hoi_segments_for_hand(self, hand_key: str) -> List[Dict]:
        segs = []
        for ev in self.events:
            h = ev.get("hoi_data", {}).get(hand_key, {})
            s = h.get("interaction_start")
            e = h.get("interaction_end")
            if s is None or e is None:
                continue
            onset = h.get("functional_contact_onset", s)
            verb = h.get("verb", "")
            base_color = self._hoi_color_for_verb(verb)
            segs.append(
                {
                    "event_id": ev.get("event_id"),
                    "start": int(s),
                    "end": int(e),
                    "onset": int(onset) if onset is not None else int(s),
                    "verb": verb,
                    "color": base_color,
                    "auto_color": base_color is None,
                }
            )
        segs = sorted(segs, key=lambda x: (x["start"], x.get("event_id", -1)))
        palette = list(PRESET_COLORS.values()) or [QColor(120, 120, 120)]
        prev_color = None
        for idx, seg in enumerate(segs):
            auto_color = bool(seg.get("auto_color", False))
            color = seg.get("color")
            if auto_color:
                start_idx = seg.get("event_id")
                if start_idx is None:
                    start_idx = idx
                try:
                    start_idx = abs(int(start_idx))
                except Exception:
                    start_idx = idx
                for offset in range(len(palette)):
                    cand = palette[(start_idx + offset) % len(palette)]
                    if prev_color is None or cand.name() != prev_color.name():
                        color = cand
                        break
            elif color is None:
                color = QColor(120, 120, 120)
            seg["color"] = color
            prev_color = color
        return segs

    def _hand_has_segment(self, ev: dict, hand_key: str) -> bool:
        h = ev.get("hoi_data", {}).get(hand_key, {})
        s = h.get("interaction_start")
        e = h.get("interaction_end")
        return s is not None and e is not None

    def _on_hoi_timeline_delete(self, event_id: int, hand_key: str):
        ev = self._find_event_by_id(event_id)
        if not ev:
            return
        self._push_undo()
        ev.get("hoi_data", {})[hand_key] = self._blank_hand_data()
        self._sync_event_frames(ev)

        if not any(
            self._hand_has_segment(ev, actor["id"]) for actor in self.actors_config
        ):
            if ev in self.events:
                self.events.remove(ev)
            if self.selected_event_id == event_id:
                self.selected_event_id = None
                self.selected_hand_label = None
                self._reset_event_draft()
                if getattr(self, "hoi_timeline", None):
                    self.hoi_timeline.set_selected(None, None)
        else:
            if (
                self.selected_event_id == event_id
                and self.selected_hand_label == hand_key
            ):
                # Find 'other' actor(s) for swap logic
                others = [a["id"] for a in self.actors_config if a["id"] != hand_key]
                other = others[0] if others else hand_key
                if self._hand_has_segment(ev, other):
                    self._set_selected_event(event_id, other)
                else:
                    self.selected_hand_label = None
                    self._reset_event_draft()

        self._update_status_label()
        self._refresh_events()
        self._update_hoi_titles()
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.refresh()
        self._log("hoi_segment_delete", event_id=event_id, hand=hand_key)

    def _get_actor_short_label(self, actor_id: str) -> str:
        """Helper for timeline titles."""
        for a in self.actors_config:
            if a["id"] == actor_id:
                return a.get("short", actor_id[:1])
        return actor_id[:1]

    def _get_actor_full_label(self, actor_id: str) -> str:
        for a in self.actors_config:
            if a["id"] == actor_id:
                return a.get("label", actor_id)
        return actor_id

    def _hoi_title_for_hand(self, hand_key: str) -> str:
        base = self._get_actor_full_label(hand_key)
        if self.selected_event_id is None:
            return f"{base}\nIdle"
        ev = self._find_event_by_id(self.selected_event_id)
        if not ev:
            return f"{base}\nIdle"
        h_data = {}
        if isinstance(getattr(self, "event_draft", None), dict):
            h_data = self.event_draft.get(hand_key, {}) or {}
        if not h_data:
            h_data = ev.get("hoi_data", {}).get(hand_key, {}) or {}
        has_data = any(
            [
                h_data.get("interaction_start") is not None,
                h_data.get("interaction_end") is not None,
                h_data.get("functional_contact_onset") is not None,
                bool(h_data.get("verb")),
                h_data.get("instrument_object_id") is not None,
                h_data.get("target_object_id") is not None,
            ]
        )
        if not has_data:
            return f"{base}\nIdle"
        missing = []
        if h_data.get("interaction_start") is None or h_data.get("interaction_end") is None:
            missing.append("time")
        if h_data.get("functional_contact_onset") is None:
            missing.append("onset")
        if not h_data.get("verb"):
            missing.append("verb")
        if h_data.get("instrument_object_id") is None:
            missing.append("inst")
        if h_data.get("target_object_id") is None:
            missing.append("target")
        if missing:
            status = "Missing " + "/".join(missing[:2])
        else:
            verb = str(h_data.get("verb") or "").strip()
            status = verb if verb else "Ready"
        return f"{base}\n{status}"
    def _update_hoi_titles(self):
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.update_titles()

    def _current_hand_meta(self) -> dict:
        return {
            "verb": self.combo_verb.currentText(),
            "instrument_object_id": self.combo_instrument.currentData(),
            "target_object_id": self.combo_target.currentData(),
            "anomaly_label": self._selected_anomaly_label(),
        }

    def _ensure_hand_annotation_state(self, hand_data: Optional[dict]) -> dict:
        if not isinstance(hand_data, dict):
            hand_data = {}
        return ensure_hand_annotation_state(hand_data)

    def _hydrate_hand_annotation_state(
        self, hand_data: Optional[dict], default_source: str = "loaded_annotation"
    ) -> dict:
        if not isinstance(hand_data, dict):
            hand_data = {}
        return hydrate_existing_field_state(hand_data, default_source=default_source)

    def _set_pending_field_source(self, field_name: str, source: str) -> None:
        field_name = str(field_name or "").strip()
        if field_name:
            self._pending_field_sources[field_name] = str(source or "").strip() or "manual_ui"

    def _consume_pending_field_source(self, field_name: str, default: str) -> str:
        field_name = str(field_name or "").strip()
        if not field_name:
            return default
        source = self._pending_field_sources.pop(field_name, None)
        return str(source or default).strip() or default

    def _set_hand_field_state(
        self,
        hand_data: Optional[dict],
        field_name: str,
        *,
        source: str,
        value: Any = _UNSET,
        status: str = "confirmed",
        note: str = "",
    ) -> None:
        if not isinstance(hand_data, dict):
            return
        set_field_confirmation(
            hand_data,
            field_name,
            source=source,
            value=value,
            status=status,
            note=note,
        )

    def _suggest_hand_field(
        self,
        hand_data: Optional[dict],
        field_name: str,
        value: Any,
        *,
        source: str,
        confidence: Optional[float] = None,
        reason: str = "",
        safe_to_apply: bool = True,
    ) -> None:
        if not isinstance(hand_data, dict):
            return
        set_field_suggestion(
            hand_data,
            field_name,
            value,
            source=source,
            confidence=confidence,
            reason=reason,
            safe_to_apply=safe_to_apply,
        )

    def _clear_hand_field(self, hand_data: Optional[dict], field_name: str, source: str = "manual_clear") -> None:
        if not isinstance(hand_data, dict):
            return
        clear_field_value(hand_data, field_name, source=source)

    def _box_source(self, box: Optional[dict], default: str = "unknown_box") -> str:
        if not isinstance(box, dict):
            return default
        return str(box.get("source") or default).strip() or default

    def _compute_sparse_evidence_state(self, hand_data: Optional[dict]) -> dict:
        if not isinstance(hand_data, dict):
            return {}
        point_defs = (
            ("start", "interaction_start", "Start"),
            ("onset", "functional_contact_onset", "Onset"),
            ("end", "interaction_end", "End"),
        )
        role_defs = (
            ("instrument", "instrument_object_id", "Instrument"),
            ("target", "target_object_id", "Target"),
        )
        states = {}
        for role_slug, object_key, role_label in role_defs:
            obj_id = hand_data.get(object_key)
            obj_name = self._object_name_for_id(obj_id, default_for_none="", fallback="")
            for point_slug, time_key, time_label in point_defs:
                slot = f"{role_slug}_{point_slug}"
                frame = hand_data.get(time_key)
                base = {
                    "slot": slot,
                    "role": role_slug,
                    "role_label": role_label,
                    "time_key": point_slug,
                    "time_label": time_label,
                    "frame": _safe_int(frame),
                    "object_id": obj_id,
                    "object_name": obj_name,
                }
                if obj_id is None or frame is None:
                    states[slot] = {
                        **base,
                        "status": "blocked",
                        "source": "missing_context",
                        "note": "Sparse evidence becomes active once both the object link and keyframe exist.",
                    }
                    continue
                try:
                    frame_int = int(frame)
                except Exception:
                    states[slot] = {
                        **base,
                        "status": "blocked",
                        "source": "invalid_frame",
                        "note": "The keyframe is not valid yet.",
                    }
                    continue
                match = None
                for box in self.bboxes.get(frame_int, []) or []:
                    if box.get("id") == obj_id:
                        match = box
                        break
                if match is not None:
                    states[slot] = {
                        **base,
                        "status": "confirmed",
                        "source": self._box_source(match, default="box_observed"),
                        "note": f"{role_label} evidence observed on the {time_label.lower()} keyframe.",
                    }
                else:
                    states[slot] = {
                        **base,
                        "status": "missing",
                        "source": "missing_evidence",
                        "note": f"No {role_slug} box was found on the {time_label.lower()} keyframe.",
                    }
        hand_data["_sparse_evidence_state"] = copy.deepcopy(states)
        return states

    def _sparse_evidence_summary(self, hand_data: Optional[dict]) -> dict:
        state = self._compute_sparse_evidence_state(hand_data)
        expected = 0
        confirmed = 0
        missing = 0
        blocked = 0
        for row in list(state.values()):
            status = str((row or {}).get("status") or "").strip().lower()
            if status == "blocked":
                blocked += 1
                continue
            expected += 1
            if status == "confirmed":
                confirmed += 1
            elif status == "missing":
                missing += 1
        return {
            "expected": expected,
            "confirmed": confirmed,
            "missing": missing,
            "blocked": blocked,
            "state": state,
        }

    def _refresh_sparse_evidence_snapshots(self) -> None:
        for event in list(self.events or []):
            hoi_data = event.get("hoi_data", {}) or {}
            if not isinstance(hoi_data, dict):
                continue
            for actor in self.actors_config:
                hand_key = actor["id"]
                hand_data = hoi_data.get(hand_key)
                if isinstance(hand_data, dict):
                    self._compute_sparse_evidence_state(hand_data)
        for hand_key, hand_data in list((getattr(self, "event_draft", {}) or {}).items()):
            if isinstance(hand_data, dict):
                self._compute_sparse_evidence_state(hand_data)

    def _sync_event_videomae_suggestions(
        self, event_id: int, event: Optional[dict], candidates: List[dict]
    ) -> None:
        if not event or not candidates:
            return
        top = dict(candidates[0] or {})
        label = str(top.get("label") or "").strip()
        if not label:
            return
        confidence = top.get("score")
        reason = "VideoMAE top-1 candidate for the current action segment."
        for actor in self.actors_config:
            hand_key = actor["id"]
            hand_data = event.get("hoi_data", {}).get(hand_key, {}) or {}
            self._ensure_hand_annotation_state(hand_data)
            state = get_field_state(hand_data, "verb")
            current = str(hand_data.get("verb") or "").strip()
            if state.get("status") == "confirmed" and current:
                continue
            self._suggest_hand_field(
                hand_data,
                "verb",
                label,
                source="videomae_top1",
                confidence=confidence,
                reason=reason,
                safe_to_apply=True,
            )
            if self.selected_event_id == event_id and hand_key in self.event_draft:
                draft = self.event_draft.get(hand_key, {})
                self._ensure_hand_annotation_state(draft)
                self._suggest_hand_field(
                    draft,
                    "verb",
                    label,
                    source="videomae_top1",
                    confidence=confidence,
                    reason=reason,
                    safe_to_apply=True,
                )

    def _blank_hand_data(self) -> dict:
        hand_data = {
            "verb": "",
            "instrument_object_id": None,
            "target_object_id": None,
            "interaction_start": None,
            "functional_contact_onset": None,
            "interaction_end": None,
            "anomaly_label": self.extra_label_config.get("default_label", "Normal"),
        }
        return self._ensure_hand_annotation_state(hand_data)

    def _on_hoi_timeline_select(self, event_id: int, hand_key: str):
        self._set_selected_event(event_id, hand_key)
        self._focus_inspector_tab("event")

    def _on_hoi_timeline_update(
        self, event_id: int, hand_key: str, start: int, end: int, onset: int
    ):
        ev = self._find_event_by_id(event_id)
        if not ev:
            return
        self._push_undo()
        h = ev.get("hoi_data", {}).setdefault(hand_key, self._blank_hand_data())
        self._ensure_hand_annotation_state(h)
        h["interaction_start"] = int(start)
        h["interaction_end"] = int(end)
        h["functional_contact_onset"] = int(onset)
        self._set_hand_field_state(
            h, "interaction_start", source="manual_timeline", status="confirmed"
        )
        self._set_hand_field_state(
            h, "interaction_end", source="manual_timeline", status="confirmed"
        )
        self._set_hand_field_state(
            h,
            "functional_contact_onset",
            source="manual_timeline",
            status="confirmed",
        )
        if h.get("anomaly_label") in (None, ""):
            h["anomaly_label"] = self.extra_label_config.get("default_label", "Normal")
        self._sync_event_frames(ev)
        if self.selected_event_id == event_id:
            self.event_draft[hand_key] = dict(h)
            self._update_status_label()
        self._refresh_events()
        self.hoi_timeline.refresh()
        self._invalidate_videomae_candidates(event_id)
        if self.selected_event_id == event_id:
            self._update_action_top5_display(event_id)
            self._queue_action_label_refresh(event_id, delay_ms=450)
        self._log(
            "hoi_event_edit_frames",
            event_id=event_id,
            hand=hand_key,
            start=start,
            onset=onset,
            end=end,
        )

    def _on_hoi_timeline_create(
        self, hand_key: str, start: int, end: int, onset: int
    ) -> Optional[int]:
        if not self.player.cap:
            QMessageBox.information(
                self, "Info", "Load a video before creating HandOI segments."
            )
            return None
        self._push_undo()
        new_event = {
            "event_id": self.event_id_counter,
            "frames": [start, end],
            "hoi_data": {actor["id"]: self._blank_hand_data() for actor in self.actors_config},
        }
        hand_data = new_event["hoi_data"][hand_key]
        hand_data.update(self._current_hand_meta())
        hand_data["interaction_start"] = int(start)
        hand_data["interaction_end"] = int(end)
        hand_data["functional_contact_onset"] = int(onset)
        self._set_hand_field_state(
            hand_data, "interaction_start", source="manual_timeline", status="confirmed"
        )
        self._set_hand_field_state(
            hand_data, "interaction_end", source="manual_timeline", status="confirmed"
        )
        self._set_hand_field_state(
            hand_data,
            "functional_contact_onset",
            source="manual_timeline",
            status="confirmed",
        )
        if str(hand_data.get("verb") or "").strip():
            self._set_hand_field_state(
                hand_data, "verb", source="manual_create", status="confirmed"
            )
        if hand_data.get("instrument_object_id") is not None:
            self._set_hand_field_state(
                hand_data,
                "instrument_object_id",
                source="manual_create",
                status="confirmed",
            )
        if hand_data.get("target_object_id") is not None:
            self._set_hand_field_state(
                hand_data,
                "target_object_id",
                source="manual_create",
                status="confirmed",
            )
        if hand_data.get("anomaly_label") in (None, ""):
            hand_data["anomaly_label"] = self.extra_label_config.get("default_label", "Normal")
        self._sync_event_frames(new_event)
        self.events.append(new_event)
        self.event_id_counter += 1
        self._refresh_events()
        self.hoi_timeline.refresh()
        self._set_selected_event(new_event["event_id"], hand_key)
        self._log(
            "hoi_event_create",
            event_id=new_event["event_id"],
            hand=hand_key,
            start=start,
            onset=onset,
            end=end,
        )
        return new_event["event_id"]

    def _jump_to_selected_start(self):
        self._jump_to_selected_keyframe("interaction_start")

    def _jump_to_selected_onset(self):
        self._jump_to_selected_keyframe("functional_contact_onset")

    def _jump_to_selected_end(self):
        self._jump_to_selected_keyframe("interaction_end")

    def _jump_to_selected_keyframe(self, key: str):
        if self.selected_event_id is None or not self.selected_hand_label:
            return
        ev = self._find_event_by_id(self.selected_event_id)
        if not ev:
            return
        hand_data = ev.get("hoi_data", {}).get(self.selected_hand_label, {})
        frame = hand_data.get(key)
        if frame is None:
            return
        try:
            self.player.seek(int(frame))
            self._refresh_boxes_for_frame(self.player.current_frame)
            self._update_overlay(self.player.current_frame)
        except Exception:
            pass

    def _on_hoi_timeline_hover(self, frame: int):
        if not self.player.cap:
            return
        if getattr(self.player, "is_playing", False):
            return
        if frame is None or frame < 0:
            target = int(getattr(self.player, "current_frame", 0))
            try:
                self.player.seek(target, preview_only=True)
            except Exception:
                return
            self._refresh_boxes_for_frame(target)
            self._update_overlay(target)
            return
        try:
            target = max(self.player.crop_start, min(frame, self.player.crop_end))
        except Exception:
            target = int(frame)
        self.player.seek(target, preview_only=True)
        self._refresh_boxes_for_frame(target)
        self._update_overlay(target)

    def _on_hoi_meta_changed(self):
        self._sync_action_panel_selection()
        if not self.selected_hand_label:
            return

        prev = dict(self.event_draft.get(self.selected_hand_label, {}) or {})
        self._save_ui_to_hand_draft(self.selected_hand_label)
        hand_data = self.event_draft.get(self.selected_hand_label, {}) or {}
        self._ensure_hand_annotation_state(hand_data)
        for field_name, default_source in (
            ("verb", "manual_ui"),
            ("instrument_object_id", "manual_link"),
            ("target_object_id", "manual_link"),
        ):
            old_value = prev.get(field_name)
            new_value = hand_data.get(field_name)
            if old_value == new_value:
                continue
            if field_name == "verb":
                is_empty = not str(new_value or "").strip()
            else:
                is_empty = new_value is None
            if is_empty:
                self._clear_hand_field(
                    hand_data,
                    field_name,
                    source=self._consume_pending_field_source(field_name, "manual_clear"),
                )
            else:
                self._set_hand_field_state(
                    hand_data,
                    field_name,
                    source=self._consume_pending_field_source(field_name, default_source),
                    status="confirmed",
                )
        if self.selected_event_id is not None:
            self._apply_draft_to_selected_event()
            self._refresh_events()
            self._update_hoi_titles()
            if getattr(self, "hoi_timeline", None):
                self.hoi_timeline.refresh()
        self._update_status_label()

    def _after_events_loaded(self):
        self.selected_event_id = None
        self.selected_hand_label = None
        self._reset_event_draft()
        self._clear_videomae_action_runtime(clear_event_scores=True)
        self._update_action_top5_display(None)
        self._update_next_best_query_panel()
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.set_selected(None, None)
            self._update_hoi_titles()
            self.hoi_timeline.refresh()

    def _on_verb_added(self, lb):
        """lb may be a LabelDef (from LabelPanel) or a name string."""
        missing = self._missing_requirements(for_relation=False)
        if missing:
            QMessageBox.information(
                self,
                "Info",
                f"Please load required data before adding verbs: {', '.join(missing)}",
            )
            return
        max_id = max([v.id for v in self.verbs], default=-1)
        if isinstance(lb, LabelDef):
            name = lb.name
        else:
            name = str(lb)
        color_name = (
            getattr(lb, "color_name", None) if isinstance(lb, LabelDef) else None
        )
        if not color_name:
            color_name = "Auto"
        new_lb = LabelDef(name=name, id=max_id + 1, color_name=color_name)
        self.verbs.append(new_lb)
        self._renumber_verbs()
        self._update_verb_combo()
        self._log("hoi_verb_add", name=new_lb.name, id=new_lb.id)

    def _on_verb_removed(self, idx: int):
        removed = None
        if 0 <= idx < len(self.verbs):
            removed = self.verbs.pop(idx)
        self._renumber_verbs()
        self.current_verb_idx = min(self.current_verb_idx, len(self.verbs) - 1)
        self._update_verb_combo()
        if removed:
            self._log("hoi_verb_remove", name=removed.name, id=removed.id)

    def _on_verb_renamed(self, idx: int, new: str):
        old = None
        if 0 <= idx < len(self.verbs):
            old = self.verbs[idx].name
            self.verbs[idx].name = new
        self._renumber_verbs()
        self._update_verb_combo()
        if old is not None:
            self._replace_verb_in_hoi_data(self.event_draft, old, new)
            for ev in self.events:
                self._replace_verb_in_hoi_data(ev.get("hoi_data", {}), old, new)
            self._refresh_events()
            self._update_hoi_titles()
            if getattr(self, "hoi_timeline", None):
                self.hoi_timeline.refresh()
            self._log("hoi_verb_rename", old=old, new=new)

    def _on_verb_selected(self, idx: int):
        if idx < 0 or idx >= len(self.verbs) or self._action_panel_sync:
            return
        self._sync_action_panel_selection(self.verbs[idx].name)

    def _apply_verb_choice(self, verb_name: str, source: str = "manual"):
        verb_name = str(verb_name or "").strip()
        if not verb_name:
            return
        current = self.combo_verb.currentText().strip()
        self._set_pending_field_source("verb", source)
        if current != verb_name:
            self.combo_verb.setCurrentText(verb_name)
        else:
            self._sync_action_panel_selection(verb_name)
            self._on_hoi_meta_changed()
        self._log(
            "hoi_apply_verb_choice",
            source=source,
            verb=verb_name,
            event_id=self.selected_event_id,
            hand=self.selected_hand_label,
        )

    def _sync_action_panel_selection(self, verb_name: str = None):
        if self._action_panel_sync:
            return
        verb_name = str(verb_name if verb_name is not None else self.combo_verb.currentText()).strip()
        self._action_panel_sync = True
        try:
            if hasattr(self, "label_panel") and self.label_panel:
                if verb_name and self.label_panel.index_of_label(verb_name) >= 0:
                    self.label_panel.select_label_by_name(verb_name)
                else:
                    self.label_panel.clear_selection()
        finally:
            self._action_panel_sync = False

    def _on_verb_library_item_clicked(self, item: QListWidgetItem):
        if item is None:
            return
        verb_name = str(item.data(Qt.UserRole) or item.text() or "").strip()
        if not verb_name or verb_name == "__OTHER_LABEL__":
            return
        self._apply_verb_choice(verb_name, source="verb_library")

    def _on_verb_double_clicked(self, item: QListWidgetItem):
        self._on_verb_library_item_clicked(item)

    def _on_top5_item_clicked(self, item: QListWidgetItem):
        if item is None:
            return
        verb_name = str(item.data(Qt.UserRole) or "").strip()
        if not verb_name:
            return
        self._apply_verb_choice(verb_name, source="videomae_top5")

    def _renumber_verbs(self):
        """Ensure ids are monotonically increasing and new ids append after max."""
        cur_max = 0
        for v in self.verbs:
            cur_max = max(cur_max, getattr(v, "id", 0))
        # ensure unique increasing ids
        seen = set()
        for v in self.verbs:
            if v.id in seen:
                cur_max += 1
                v.id = cur_max
            seen.add(v.id)
        # advance next id in UI
        try:
            self.label_panel.id_spin.setValue(cur_max + 1)
        except Exception:
            pass

    def _normalize_hand_label(self, label: str) -> Optional[str]:
        if label is None:
            return None
        norm = str(label).strip().lower().replace(" ", "_")
        for actor in self.actors_config:
            aid = actor["id"]
            if norm == aid.lower():
                return aid
            if norm in (
                actor.get("short", "").lower(),
                actor.get("label", "").lower(),
            ):
                return aid
        # Fallbacks for very common terms
        if norm in ("left", "l_hand"):
            for actor in self.actors_config:
                if "left" in actor["id"].lower():
                    return actor["id"]
            return self.actors_config[0]["id"]
        if norm in ("right", "r_hand"):
            for actor in self.actors_config:
                if "right" in actor["id"].lower():
                    return actor["id"]
            return self.actors_config[1]["id"] if len(self.actors_config) > 1 else self.actors_config[0]["id"]
        return None

    def _is_hand_label(self, label: str) -> bool:
        if not label:
            return False
        l_str = str(label).strip()
        return any(actor["id"] == l_str for actor in self.actors_config)

    def _color_for_index(self, idx: int) -> str:
        """Pick a non-gray preset color cycling through the palette."""
        palette = [k for k in PRESET_COLORS.keys() if k.lower() != "gray"] or list(
            PRESET_COLORS.keys()
        )
        if not palette:
            return "gray"
        return palette[idx % len(palette)]

    def _verb_color(self, name: str) -> str:
        """Return a QColor-friendly string for a verb name."""
        for v in self.verbs:
            if v.name == name:
                return color_from_key(getattr(v, "color_name", "gray")).name()
        return "#f08c00"

    def _missing_requirements(self, for_relation: bool = False) -> List[str]:
        """
        New check chain: only checks Video and Hands.
        Does not check classes or object bboxes.
        """
        missing = []
        if not self.player.cap:
            missing.append("video")

        # Only check for hands data when creating a relation
        if for_relation:
            # Check if any hand data exists
            has_hand = any(
                b
                for b in self.raw_boxes
                if self._is_hand_label(b.get("label"))
            )
            if not has_hand:
                missing.append("hand bboxes (XML)")

        return missing

    # ---------- undo/redo ----------
    def _snapshot_state(self) -> dict:
        """Capture a deep copy of event + bbox state for undo/redo."""
        return {
            "events": copy.deepcopy(self.events),
            "event_id_counter": self.event_id_counter,
            "event_draft": copy.deepcopy(self.event_draft),
            "selected_hand_label": self.selected_hand_label,
            "selected_event_id": self.selected_event_id,
            "raw_boxes": copy.deepcopy(self.raw_boxes),
            "box_id_counter": self.box_id_counter,
            "global_object_map": copy.deepcopy(self.global_object_map),
            "id_to_category": copy.deepcopy(self.id_to_category),
            "object_id_counter": self.object_id_counter,
            "class_map": copy.deepcopy(self.class_map),
            "instrument_selected": self.combo_instrument.currentData(),
            "target_selected": self.combo_target.currentData(),
            "current_frame": int(getattr(self.player, "current_frame", 0)),
        }

    def _restore_state(self, state: dict):
        """Restore a snapshot created by _snapshot_state."""
        self.events = copy.deepcopy(state.get("events", []))
        self.event_id_counter = state.get("event_id_counter", 0)
        self.event_draft = copy.deepcopy(state.get("event_draft", {}))
        self.selected_hand_label = state.get("selected_hand_label", None)
        self.selected_event_id = state.get("selected_event_id", None)
        self.raw_boxes = copy.deepcopy(state.get("raw_boxes", []))
        self.box_id_counter = state.get("box_id_counter", 1)
        self.global_object_map = copy.deepcopy(state.get("global_object_map", {}))
        self.id_to_category = copy.deepcopy(state.get("id_to_category", {}))
        self.object_id_counter = state.get("object_id_counter", 0)
        self.class_map = copy.deepcopy(state.get("class_map", {}))

        self._rebuild_bboxes_from_raw()

        inst_sel = state.get("instrument_selected")
        tgt_sel = state.get("target_selected")
        self._rebuild_object_combos(inst_sel, tgt_sel)

        frame = state.get("current_frame", self.player.current_frame)
        if self.player.cap:
            self.player.seek(frame)
        self._refresh_boxes_for_frame(frame)
        self._set_frame_controls(frame)

        for aid, chk in self.actor_controls.items():
            chk.blockSignals(True)
            chk.setChecked(self.selected_hand_label == aid)
            chk.blockSignals(False)

        if self.selected_hand_label:
            self._load_hand_draft_to_ui(self.selected_hand_label)
        else:
            self._update_status_label()
        self.list_objects.setEnabled(self.selected_hand_label is not None)
        for ev in self.events:
            self._sync_event_frames(ev)
        self._refresh_events()
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.set_selected(
                self.selected_event_id, self.selected_hand_label
            )
            self.hoi_timeline.set_current_frame(frame)
            self.hoi_timeline.refresh()
        
        self._update_action_top5_display(self.selected_event_id)

    def _push_undo(self):
        if self._undo_block:
            return
        if self.validation_enabled:
            self.validation_modified = True
            self.validation_change_count += 1
        self._hoi_undo_stack.append(self._snapshot_state())
        if len(self._hoi_undo_stack) > self._undo_limit:
            self._hoi_undo_stack.pop(0)
        self._hoi_redo_stack.clear()

    def _hoi_undo(self):
        if not self._hoi_undo_stack:
            return
        self._undo_block = True
        try:
            current = self._snapshot_state()
            state = self._hoi_undo_stack.pop()
            self._hoi_redo_stack.append(current)
            self._restore_state(state)
        finally:
            self._undo_block = False

    def _hoi_redo(self):
        if not self._hoi_redo_stack:
            return
        self._undo_block = True
        try:
            current = self._snapshot_state()
            state = self._hoi_redo_stack.pop()
            self._hoi_undo_stack.append(current)
            self._restore_state(state)
        finally:
            self._undo_block = False

    def _clear_undo_history(self):
        self._hoi_undo_stack.clear()
        self._hoi_redo_stack.clear()

    def set_oplog_enabled(self, enabled: bool) -> None:
        self.set_logging_policy(
            bool(enabled), bool(getattr(self, "validation_summary_enabled", True))
        )

    def set_logging_policy(
        self, oplog_enabled: bool, validation_summary_enabled: bool
    ) -> None:
        if getattr(self, "op_logger", None) is not None:
            self.op_logger.enabled = bool(oplog_enabled)
        if getattr(self, "validation_op_logger", None) is not None:
            self.validation_op_logger.enabled = bool(oplog_enabled)
        self.validation_summary_enabled = bool(validation_summary_enabled)

    def _flush_ops_log_safely(self, logger, log_path: str, context: str) -> None:
        if logger is None or not log_path:
            return
        should_write = bool(getattr(logger, "enabled", False))
        try:
            logger.flush(log_path)
            if should_write:
                print(f"[LOG] {log_path}")
        except Exception as ex:
            print(f"[LOG][ERROR] {context} ops log write failed: {log_path} ({ex})")
            QMessageBox.warning(
                self,
                "Operation log",
                f"Operation log write failed, but annotation save already succeeded.\n\nTarget: {log_path}\n\n{ex}",
            )

    def _log(self, event: str, **fields):
        # Keep the log focused on user-driven actions.
        auto_events = {"frame_advanced"}
        if event in auto_events:
            return
        fields.setdefault("assist_mode", self._experiment_mode_key())
        fields.setdefault("selected_event_id", self.selected_event_id)
        fields.setdefault("selected_hand", self.selected_hand_label or "")
        try:
            base = os.path.splitext(os.path.basename(self.video_path or ""))[0]
        except Exception:
            base = ""
        if base:
            fields.setdefault("video_id", base)
        if getattr(self, "op_logger", None):
            try:
                frame = getattr(self.player, "current_frame", None)
            except Exception:
                frame = None
            if frame is not None:
                fields.setdefault("frame", frame)
            if self.validation_enabled and self.validator_name:
                fields.setdefault("validator", self.validator_name)
            self.op_logger.log(event, **fields)
        if self.validation_enabled and getattr(self, "validation_op_logger", None):
            self.validation_op_logger.log(event, **fields)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Left:
            self._seek_relative(-1)
            event.accept()
            return
        if key == Qt.Key_Right:
            self._seek_relative(+1)
            event.accept()
            return
        if key == Qt.Key_Up:
            self._seek_seconds(-1)
            event.accept()
            return
        if key == Qt.Key_Down:
            self._seek_seconds(+1)
            event.accept()
            return
        return super().keyPressEvent(event)

    # ---------- IO ----------
    @staticmethod
    def _normalized_video_path(path: str) -> str:
        raw = str(path or "").strip()
        if not raw:
            return ""
        try:
            return os.path.normcase(os.path.abspath(raw))
        except Exception:
            return os.path.normcase(raw)

    def _hoi_has_annotation_data(self) -> bool:
        return bool(self.events or self.raw_boxes)

    def _apply_video_session(
        self,
        path: str,
        start: int = None,
        end: int = None,
        frame: int = None,
        log_event: str = "hoi_load_video",
    ) -> bool:
        target_path = str(path or "").strip()
        if not target_path:
            return False
        target_norm = self._normalized_video_path(target_path)
        current_norm = self._normalized_video_path(self.video_path)
        reuse_loaded = bool(self.player.cap) and bool(current_norm) and current_norm == target_norm
        if not reuse_loaded:
            if not self.player.load(target_path):
                return False

        total_frames = int(getattr(self.player, "frame_count", 0) or 0)
        if total_frames <= 0:
            return False
        start_frame = 0 if start is None else int(start)
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = total_frames - 1 if end is None else int(end)
        end_frame = max(start_frame, min(end_frame, total_frames - 1))
        current_frame = start_frame if frame is None else int(frame)
        current_frame = max(start_frame, min(current_frame, end_frame))

        self.player.set_crop(start_frame, end_frame)
        if int(getattr(self.player, "current_frame", start_frame)) != current_frame:
            self.player.seek(current_frame)

        self.video_path = target_path
        self.start_offset = start_frame
        self.end_frame = end_frame

        self.spin_start_offset.blockSignals(True)
        self.spin_start_offset.setMaximum(max(0, total_frames - 1))
        self.spin_start_offset.setValue(start_frame)
        self.spin_start_offset.blockSignals(False)

        self.spin_end_frame.blockSignals(True)
        self.spin_end_frame.setMaximum(max(0, total_frames - 1))
        self.spin_end_frame.setValue(end_frame)
        self.spin_end_frame.blockSignals(False)

        self.spin_jump.setMinimum(start_frame)
        self.spin_jump.setMaximum(max(start_frame, end_frame))
        self.spin_jump.setValue(current_frame)

        self.slider.blockSignals(True)
        self.slider.setMinimum(start_frame)
        self.slider.setMaximum(end_frame)
        self.slider.setValue(current_frame)
        self.slider.blockSignals(False)

        self._rebuild_bboxes_from_raw()
        self._refresh_boxes_for_frame(current_frame)
        self._set_frame_controls(current_frame)

        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.set_frame_count(self.player.frame_count)
            self.hoi_timeline.set_current_frame(current_frame)
            self.hoi_timeline.refresh()

        self._update_play_pause_button()
        if log_event:
            self._log(
                log_event,
                path=target_path,
                start=start_frame,
                end=end_frame,
                frame=current_frame,
                frames=self.player.frame_count,
            )
        return True

    def inherit_action_video_session(self, session: dict) -> bool:
        if not isinstance(session, dict):
            return False
        path = str(session.get("path") or "").strip()
        if not path:
            return False
        start = int(session.get("start", 0) or 0)
        end = int(session.get("end", start) or start)
        frame = int(session.get("frame", start) or start)
        target_norm = self._normalized_video_path(path)
        current_norm = self._normalized_video_path(self.video_path)
        has_annotations = self._hoi_has_annotation_data()

        if has_annotations:
            if current_norm and current_norm == target_norm and self.player.cap:
                target_frame = max(
                    int(getattr(self.player, "crop_start", 0)),
                    min(frame, int(getattr(self.player, "crop_end", frame))),
                )
                self.player.seek(target_frame)
                self._refresh_boxes_for_frame(target_frame)
                self._set_frame_controls(target_frame)
                if getattr(self, "hoi_timeline", None):
                    self.hoi_timeline.set_current_frame(target_frame)
                    self.hoi_timeline.refresh()
                self._update_play_pause_button()
                self._log("hoi_sync_frame_from_action", path=path, frame=target_frame)
                return True
            return False

        ok = self._apply_video_session(
            path,
            start=start,
            end=end,
            frame=frame,
            log_event="hoi_inherit_action_video",
        )
        if ok:
            self._mark_hoi_saved()
        return ok

    def _load_video(self):
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Load Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )
        if not fp:
            return
        if not self._apply_video_session(fp, log_event="hoi_load_video"):
            QMessageBox.warning(self, "Error", "Failed to load video.")
            return

    def _load_bboxes(self):
        """
        Phase 1 Smart Loader (Corrected):
        Saves filtered YOLO boxes into self.raw_boxes to prevent overwriting.
        """
        if not self._guard_experiment_mode("detection"):
            return
        # --- 1. Dependency Check ---
        if not self.player.cap:
            QMessageBox.warning(
                self, "Missing prerequisite", "Please load a video first (resolution needed)."
            )
            return
        if not self.global_object_map:
            QMessageBox.warning(
                self,
                "Missing prerequisite",
                "Please import Instrument/Target lists first.",
            )
            return
        if not self.class_map:
            QMessageBox.warning(
                self,
                "Missing prerequisite",
                "Please load a class map first (data.yaml).",
            )
            return

        dir_path = QFileDialog.getExistingDirectory(
            self, "Select YOLO labels directory"
        )
        if not dir_path:
            return

        # --- 2. Build Smart Map ---
        category_to_default_id = self._build_category_to_default_id()

        if not category_to_default_id:
            QMessageBox.warning(
                self,
                "Error",
                "Cannot build the default object map. Check the Instrument/Target lists.",
            )
            return

        # --- 3. Parse Loop ---
        loaded_count = 0
        w, h = self.player._frame_w, self.player._frame_h
        new_raw_boxes = []

        import os

        files = sorted([f for f in os.listdir(dir_path) if f.endswith(".txt")])

        for fname in files:
            # Get raw frame index (without offset)
            frame_idx = self._frame_from_filename(fname)

            path = os.path.join(dir_path, fname)
            with open(path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        cls_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                    except Exception:
                        continue

                    cls_name = self._norm_category(self.class_map.get(cls_id, ""))
                    if not cls_name:
                        continue

                    # Match ID
                    matched_uid = category_to_default_id.get(cls_name)

                    if matched_uid is not None:
                        # Calc absolute coordinates
                        x1 = (cx - bw / 2) * w
                        y1 = (cy - bh / 2) * h
                        x2 = (cx + bw / 2) * w
                        y2 = (cy + bh / 2) * h

                        # Construct raw_box dict
                        # Must include orig_frame for _rebuild to calculate offset correctly
                        new_raw_boxes.append(
                            {
                                "id": matched_uid,  # Forced merge to global ID
                                "orig_frame": frame_idx,
                                "label": self.class_map.get(cls_id, ""),
                                "class_id": cls_id,
                                "source": "yolo_import",
                                "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2,
                            }
                        )
                        loaded_count += 1

        # --- 4. Update Memory ---
        # Append new data to raw_boxes
        # Strategy: Keep hands (Left_hand/Right_hand), clear other old objects, add new objects
        has_existing_objects = any(
            b
            for b in self.raw_boxes
            if not self._is_hand_label(b.get("label"))
        )
        if new_raw_boxes or has_existing_objects:
            self._push_undo()
        self.raw_boxes = [
            b for b in self.raw_boxes if self._is_hand_label(b.get("label"))
        ]
        self.raw_boxes.extend(new_raw_boxes)

        # Rebuild bboxes index
        self._rebuild_bboxes_from_raw()

        self._refresh_boxes_for_frame(self.player.current_frame)
        self._log("hoi_load_bboxes", path=dir_path, count=loaded_count)
        QMessageBox.information(
            self,
            "Done",
            f"Loaded {loaded_count} detection boxes.\nOld objects cleared, hands preserved.",
        )
        self._mark_hoi_saved()

    def _load_yaml(self):
        """Load data.yaml to get class_id -> class_name mapping"""
        fp, _ = QFileDialog.getOpenFileName(
            self,
            "Load Class Map (data.yaml)",
            "",
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if not fp:
            return
        try:
            self.class_map = self._parse_yaml_names(fp)
            QMessageBox.information(
                self, "Loaded", f"Loaded {len(self.class_map)} classes from yaml."
            )
            self._log("hoi_load_classes", path=fp, count=len(self.class_map))
            self._sync_library_with_class_map()
        except Exception as ex:
            QMessageBox.warning(self, "Error", f"Failed to parse data.yaml:\n{ex}")

    def _sync_library_with_class_map(self):
        """Automatically add all classes from data.yaml to Instrument/Target libraries."""
        if not self.class_map:
            return

        added_count = 0
        for cid, class_name in self.class_map.items():
            norm_name = self._norm_category(class_name)
            if norm_name in ("left_hand", "right_hand"):
                continue

            existing_match = None
            for name in self.global_object_map.keys():
                if self._norm_category(name) == norm_name:
                    existing_match = name
                    break

            if not existing_match:
                uid = self.object_id_counter
                self.global_object_map[class_name] = uid
                self.id_to_category[class_name] = class_name
                self.object_id_counter += 1

                display_text = f"[{uid}] {class_name}"
                self.combo_instrument.addItem(display_text, uid)
                self.combo_target.addItem(display_text, uid)
                added_count += 1

        if added_count > 0:
            QMessageBox.information(
                self,
                "Sync Complete",
                f"Added {added_count} new classes from YOLO model to the object library.",
            )

    def _load_verbs_txt(self):
        """Import verb list txt with duplication check"""
        fp, _ = QFileDialog.getOpenFileName(
            self, "Import Verb List", "", "Text Files (*.txt);;All Files (*)"
        )
        if not fp:
            return
        try:
            lines = [
                ln.strip()
                for ln in open(fp, "r", encoding="utf-8").read().splitlines()
                if ln.strip()
            ]

            # Calc max ID for new assignment
            max_id = max([v.id for v in self.verbs], default=-1)
            added_count = 0

            for idx, ln in enumerate(lines):
                parts = ln.split()
                if not parts:
                    continue
                name = parts[0]

                # Check for duplicates (case-insensitive)
                if any(v.name.lower() == name.lower() for v in self.verbs):
                    continue

                max_id += 1
                color_name = "Auto"
                self.verbs.append(LabelDef(name=name, id=max_id, color_name=color_name))
                added_count += 1

            self._renumber_verbs()
            self._update_verb_combo()
            self.label_panel.refresh()

            if added_count > 0:
                QMessageBox.information(
                    self,
                    "Loaded",
                    f"Successfully imported {added_count} new verbs (duplicates skipped).",
                )
            else:
                QMessageBox.information(
                    self, "Info", "No new verbs imported (all exist)."
                )
            self._log("hoi_load_verbs", path=fp, added=added_count)

        except Exception as ex:
            QMessageBox.warning(self, "Error", f"Failed to load verbs:\n{ex}")

    def _load_yolo_model(self):
        if not self._guard_experiment_mode("detection"):
            return
        """Load Ultralytics YOLOv11 .pt weights for current-frame detection."""
        fp, _ = QFileDialog.getOpenFileName(
            self, "Load YOLO Model", "", "Weight Files (*.pt *.pth);;All Files (*)"
        )
        if not fp:
            return
        try:
            from ultralytics import YOLO
        except Exception as ex:
            QMessageBox.warning(
                self, "Missing package", f"Ultralytics is not available:\n{ex}"
            )
            return
        try:
            self.yolo_model = YOLO(fp)
            self.yolo_weights_path = fp
            QMessageBox.information(
                self, "Loaded", f"YOLO model loaded:\n{os.path.basename(fp)}"
            )
            self._log("hoi_load_yolo_model", path=fp)
        except Exception as ex:
            QMessageBox.warning(self, "Error", f"Failed to load model:\n{ex}")

    def _load_videomae_model(self):
        if not self._guard_experiment_mode("semantic"):
            return
        fp, _ = QFileDialog.getOpenFileName(
            self, "Load VideoMAE Model", "", "Weight Files (*.pt *.pth *.bin *.safetensors);;All Files (*)"
        )
        if not fp:
            return
        self.videomae_weights_path = fp
        self._videomae_loaded_key = ""
        self._clear_videomae_action_runtime(clear_event_scores=False)
        if self.videomae_verb_list_path:
            self._init_videomae()
        else:
            QMessageBox.information(self, "Info", "Model path stored. Please load a verb list to initialize.")

    def _load_videomae_verb_list(self):
        if not self._guard_experiment_mode("semantic"):
            return
        fp, _ = QFileDialog.getOpenFileName(
            self, "Load VideoMAE Verb List", "", "YAML/JSON Files (*.yaml *.json);;Text Files (*.txt);;All Files (*)"
        )
        if not fp:
            return
        self.videomae_verb_list_path = fp
        self._videomae_loaded_key = ""
        self._clear_videomae_action_runtime(clear_event_scores=False)
        if self.videomae_weights_path:
            self._init_videomae()
        else:
            QMessageBox.information(self, "Info", "Verb list stored. Please load model weights to initialize.")

    def _videomae_model_key(self) -> str:
        weights = os.path.abspath(str(self.videomae_weights_path or "").strip())
        verbs = os.path.abspath(str(self.videomae_verb_list_path or "").strip())
        if not weights or not verbs:
            return ""
        return f"{weights}|{verbs}"

    def _videomae_ready(self) -> bool:
        return bool(
            self.video_path
            and self.videomae_handler.model is not None
            and self.videomae_handler.processor is not None
            and self._videomae_loaded_key
            and self._videomae_loaded_key == self._videomae_model_key()
        )

    def _clear_videomae_action_runtime(self, clear_event_scores: bool = False):
        try:
            self._videomae_auto_timer.stop()
        except Exception:
            pass
        self._videomae_auto_event_id = None
        self._videomae_auto_force_refresh = False
        self._videomae_action_cache.clear()
        self._videomae_event_signatures.clear()
        if clear_event_scores:
            for event in self.events:
                if isinstance(event, dict):
                    event.pop("videomae_top5", None)
                    for actor in self.actors_config:
                        hand_data = event.get("hoi_data", {}).get(actor["id"], {}) or {}
                        self._ensure_hand_annotation_state(hand_data)
                        suggestion = hand_data.get("_field_suggestions", {}).get("verb", {})
                        source = str((suggestion or {}).get("source") or "").strip()
                        if source.startswith("videomae"):
                            clear_field_suggestion(hand_data, "verb")
            for actor in self.actors_config:
                draft = self.event_draft.get(actor["id"], {}) or {}
                self._ensure_hand_annotation_state(draft)
                suggestion = draft.get("_field_suggestions", {}).get("verb", {})
                source = str((suggestion or {}).get("source") or "").strip()
                if source.startswith("videomae"):
                    clear_field_suggestion(draft, "verb")

    def _normalize_videomae_candidates(self, candidates) -> List[dict]:
        rows: List[dict] = []
        for cand in list(candidates or []):
            if not isinstance(cand, dict):
                continue
            label = str(cand.get("label") or "").strip()
            if not label:
                continue
            score = cand.get("score")
            try:
                score = None if score is None else float(score)
            except Exception:
                score = None
            rows.append({"label": label, "score": score})
        return rows

    def _videomae_signature_for_event(self, event: Optional[dict]) -> str:
        if not event or not self._videomae_ready():
            return ""
        start, end = self._compute_event_frames(event)
        if start is None or end is None:
            return ""
        payload = {
            "video_path": os.path.abspath(str(self.video_path or "").strip()),
            "start": int(start),
            "end": int(end),
            "model": self._videomae_loaded_key,
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def _cached_videomae_candidates(self, event_id: int, event: Optional[dict]):
        signature = self._videomae_signature_for_event(event)
        if not signature:
            return [], ""
        rows: List[dict] = []
        if self._videomae_event_signatures.get(int(event_id)) == signature and event:
            rows = self._normalize_videomae_candidates(event.get("videomae_top5"))
        if not rows:
            rows = self._normalize_videomae_candidates(
                self._videomae_action_cache.get(signature)
            )
        return rows, signature

    def _store_videomae_candidates(
        self,
        event_id: int,
        event: Optional[dict],
        signature: str,
        candidates,
    ) -> List[dict]:
        clean = self._normalize_videomae_candidates(candidates)
        stored = [dict(row) for row in clean]
        if event is not None:
            event["videomae_top5"] = [dict(row) for row in stored]
        if signature and stored:
            self._videomae_action_cache[signature] = [dict(row) for row in stored]
            self._videomae_event_signatures[int(event_id)] = signature
        return stored

    def _invalidate_videomae_candidates(self, event_id: Optional[int]) -> None:
        if event_id is None:
            return
        try:
            key = int(event_id)
        except Exception:
            return
        self._videomae_event_signatures.pop(key, None)

    def _queue_action_label_refresh(
        self,
        event_id: Optional[int] = None,
        delay_ms: Optional[int] = None,
        force: bool = False,
    ) -> None:
        if not self._semantic_assist_enabled():
            return
        if event_id is None:
            event_id = self.selected_event_id
        if event_id is None:
            return
        try:
            event_id = int(event_id)
        except Exception:
            return
        event = self._find_event_by_id(event_id)
        if not event:
            return
        if not self._videomae_ready():
            if self.selected_event_id == event_id:
                self._update_action_top5_display(event_id)
            return
        if not force:
            cached, _signature = self._cached_videomae_candidates(event_id, event)
            if cached:
                if self.selected_event_id == event_id:
                    self._update_action_top5_display(event_id)
                return
        self._videomae_auto_event_id = event_id
        self._videomae_auto_force_refresh = bool(force)
        wait_ms = self._videomae_auto_timer.interval() if delay_ms is None else max(0, int(delay_ms))
        self._videomae_auto_timer.start(wait_ms)

    def _run_pending_action_label_refresh(self) -> None:
        event_id = self._videomae_auto_event_id
        force = bool(self._videomae_auto_force_refresh)
        self._videomae_auto_event_id = None
        self._videomae_auto_force_refresh = False
        if event_id is None or self.selected_event_id != event_id:
            return
        self._detect_action_label(
            event_id,
            interactive=False,
            auto_apply=False,
            use_cache=not force,
            force=force,
            report_errors=False,
        )

    def _init_videomae(self):
        self._clear_videomae_action_runtime(clear_event_scores=False)
        success, msg = self.videomae_handler.load_model(
            self.videomae_weights_path, self.videomae_verb_list_path
        )
        if success:
            self._videomae_loaded_key = self._videomae_model_key()
            self._sync_label_panel_with_videomae()
            QMessageBox.information(self, "Success", f"VideoMAE V2 initialized and verb list synchronized.\n{msg}")
            self._log(
                "hoi_load_videomae",
                weights=self.videomae_weights_path,
                verbs=self.videomae_verb_list_path,
            )
            if self.selected_event_id is not None:
                self._queue_action_label_refresh(self.selected_event_id, delay_ms=120)
        else:
            self._videomae_loaded_key = ""
            QMessageBox.warning(self, "Error", f"Failed to load VideoMAE V2:\n{msg}")

    def _sync_label_panel_with_videomae(self):
        if not self.videomae_handler or not self.videomae_handler.labels:
            return
        
        # Cross-reference labels and add missing ones to self.verbs
        existing_names = {v.name.lower().strip() for v in self.verbs}
        max_id = max([v.id for v in self.verbs], default=-1)
        added_count = 0
        
        for name in self.videomae_handler.labels:
            clean_name = str(name).strip()
            if not clean_name or clean_name.lower() in existing_names:
                continue
            
            max_id += 1
            self.verbs.append(LabelDef(name=clean_name, id=max_id, color_name="Auto"))
            existing_names.add(clean_name.lower())
            added_count += 1
            
        if added_count > 0:
            self._renumber_verbs()
            self._update_verb_combo()
            self.label_panel.refresh()

    def _detect_selected_action_label(self):
        if not self._guard_experiment_mode("semantic"):
            return
        if self.selected_event_id is None:
            QMessageBox.warning(self, "Warning", "Please select an action segment first.")
            return
        self._detect_action_label(
            self.selected_event_id,
            interactive=True,
            auto_apply=True,
            use_cache=True,
            force=False,
            report_errors=True,
        )

    def _refresh_selected_action_label(self):
        if not self._guard_experiment_mode("semantic"):
            return
        if self.selected_event_id is None:
            QMessageBox.warning(self, "Warning", "Please select an action segment first.")
            return
        self._detect_action_label(
            self.selected_event_id,
            interactive=False,
            auto_apply=False,
            use_cache=False,
            force=True,
            report_errors=True,
        )

    def _detect_action_label(
        self,
        event_id,
        interactive=True,
        auto_apply=True,
        use_cache=True,
        force=False,
        report_errors=False,
    ):
        notify_user = bool(interactive or report_errors)
        event = self._find_event_by_id(event_id)
        if not event:
            if notify_user:
                QMessageBox.warning(self, "Warning", "Please select an action segment first.")
            return False

        start, end = self._compute_event_frames(event)
        if start is None or end is None:
            if notify_user:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "Action segment must have start and end frames.",
                )
            return False

        signature = self._videomae_signature_for_event(event)
        candidates: List[dict] = []
        if use_cache and not force:
            candidates, signature = self._cached_videomae_candidates(event_id, event)

        if not candidates:
            if not self._videomae_ready():
                if notify_user:
                    QMessageBox.warning(
                        self,
                        "Detection Error",
                        "Load a matching VideoMAE model and verb list first.",
                    )
                return False
            raw_candidates, err = self.videomae_handler.predict(
                self.video_path, start, end
            )
            if err:
                if notify_user:
                    QMessageBox.warning(self, "Detection Error", err)
                return False
            candidates = self._normalize_videomae_candidates(raw_candidates)
            if not candidates:
                if notify_user:
                    QMessageBox.information(self, "Info", "No labels predicted.")
                return False
            if not signature:
                signature = self._videomae_signature_for_event(event)
        candidates = self._store_videomae_candidates(
            int(event_id), event, signature, candidates
        )
        self._sync_event_videomae_suggestions(int(event_id), event, candidates)
        self._update_action_top5_display(event_id)

        selected = None
        if interactive:
            dlg = ActionLabelSelector(candidates, self)
            if dlg.exec_() == QDialog.Accepted:
                selected = dlg.get_selected_label()
            else:
                return True
        elif auto_apply and candidates:
            selected = candidates[0]["label"]

        if selected:
            verb_id = -1
            for v in self.verbs:
                if v.name.lower() == selected.lower():
                    verb_id = v.id
                    break

            if verb_id != -1:
                if not interactive:
                    has_locked_confirmed = False
                    for hand in self.actors_config:
                        hid = hand["id"]
                        hand_data = event.get("hoi_data", {}).get(hid, {}) or {}
                        state = get_field_state(hand_data, "verb")
                        current_value = str(hand_data.get("verb") or "").strip()
                        if (
                            state.get("status") == "confirmed"
                            and current_value
                            and current_value != selected
                        ):
                            has_locked_confirmed = True
                            break
                    if has_locked_confirmed:
                        self._log(
                            "hoi_skip_locked_verb_autofill",
                            event_id=event_id,
                            suggested=selected,
                        )
                        return True
                event["verb_id"] = verb_id
                source_name = (
                    "videomae_selected" if interactive else "videomae_top1_auto"
                )
                target_status = "confirmed" if interactive else "suggested"
                for hand in self.actors_config:
                    hid = hand["id"]
                    if hid in event.get("hoi_data", {}):
                        event["hoi_data"][hid]["verb"] = selected
                        self._set_hand_field_state(
                            event["hoi_data"][hid],
                            "verb",
                            source=source_name,
                            status=target_status,
                        )
                    if hid in self.event_draft:
                        self.event_draft[hid]["verb"] = selected
                        self._set_hand_field_state(
                            self.event_draft[hid],
                            "verb",
                            source=source_name,
                            status=target_status,
                        )

                self._refresh_events()
                if hasattr(self, "hoi_timeline") and self.hoi_timeline:
                    self.hoi_timeline.refresh()
                if self.selected_event_id == event_id:
                    if self.selected_hand_label:
                        self._load_hand_draft_to_ui(self.selected_hand_label)
                    else:
                        self._sync_action_panel_selection(selected)
            elif notify_user:
                QMessageBox.warning(
                    self,
                    "Error",
                    f"Verb '{selected}' not found in the current project verb list.",
                )
        return True

    def _update_action_top5_display(self, event_id):
        if not hasattr(self, "label_panel") or not self.label_panel:
            return
        if not self._semantic_assist_enabled():
            self.label_panel.clear_candidate_priority()
            self._sync_action_panel_selection()
            self._update_next_best_query_panel()
            return

        if event_id is None:
            self.label_panel.clear_candidate_priority()
            self._sync_action_panel_selection()
            self._update_next_best_query_panel()
            return

        event = self._find_event_by_id(event_id)
        candidates, _signature = self._cached_videomae_candidates(int(event_id), event)
        if not event or not candidates:
            self.label_panel.clear_candidate_priority()
            self._sync_action_panel_selection()
            self._update_next_best_query_panel()
            return

        raw_candidates = [
            (cand.get("label"), cand.get("score"))
            for cand in candidates
            if cand.get("label")
        ]
        visible_candidates = [
            (name, score)
            for name, score in raw_candidates
            if self.label_panel.index_of_label(name) >= 0
        ]
        if visible_candidates:
            self.label_panel.set_candidate_priority(visible_candidates)
        else:
            self.label_panel.clear_candidate_priority()
        self._sync_action_panel_selection()
        self._update_next_best_query_panel()

    def _detect_all_action_labels(self):
        if not self._guard_experiment_mode("semantic"):
            return
        if not self.events:
            QMessageBox.information(self, "Info", "No actions to detect.")
            return
        if not self._videomae_ready():
            QMessageBox.information(
                self,
                "Info",
                "Load a VideoMAE model and verb list before batch action labeling.",
            )
            return

        progress = QProgressDialog("Detecting all action labels...", "Cancel", 0, len(self.events), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        for i, event in enumerate(self.events):
            if progress.wasCanceled():
                break
            progress.setValue(i)
            QApplication.processEvents()
            self._detect_action_label(event["event_id"], interactive=False, auto_apply=True)
        
        progress.setValue(len(self.events))
        QMessageBox.information(self, "Finished", "Batch action detection complete.")

    def _ask_existing_policy(self):
        """Ask how to handle existing boxes on the current frame."""
        box = QMessageBox(self)
        box.setWindowTitle("Existing boxes")
        box.setText(
            "This frame already has object boxes. How do you want to apply YOLO results?"
        )
        append_btn = box.addButton("Append", QMessageBox.AcceptRole)
        replace_btn = box.addButton("Replace", QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)
        remember = QCheckBox("Remember my choice")
        box.setCheckBox(remember)
        box.exec_()
        clicked = box.clickedButton()
        policy = None
        if clicked == append_btn:
            policy = "append"
        elif clicked == replace_btn:
            policy = "replace"
        if policy and remember.isChecked():
            self.yolo_existing_policy = policy
        return policy

    def _detect_current_frame_yolo(
        self,
        frame_idx=None,
        include_ids=None,
        override_policy=None,
        push_undo=True,
    ):
        """Run YOLOv11 detection on one frame and optionally filter to selected object ids."""
        if not self.player.cap:
            QMessageBox.warning(self, "Missing prerequisite", "Please load a video first.")
            return
        if not self.class_map:
            QMessageBox.warning(
                self, "Missing prerequisite", "Please load a class map first (data.yaml)."
            )
            return
        if not self.global_object_map:
            QMessageBox.warning(
                self,
                "Missing prerequisite",
                "Please import Instrument/Target lists first.",
            )
            return
        if self.yolo_model is None:
            self._load_yolo_model()
            if self.yolo_model is None:
                return

        frame_bgr = self.player.get_current_frame_bgr()
        if frame_bgr is None:
            QMessageBox.warning(self, "Error", "No video frame available.")
            return

        if frame_idx is None:
            frame_idx = int(self.player.current_frame)
        existing = [
            b
            for b in self.bboxes.get(frame_idx, [])
            if not self._is_hand_label(b.get("label"))
        ]
        replace_existing = False
        if existing:
            policy = override_policy or self.yolo_existing_policy or self._ask_existing_policy()
            if not policy:
                return
            replace_existing = policy == "replace"

        # Build category -> default object id mapping (normalized)
        category_to_default_id = self._build_category_to_default_id()

        if not category_to_default_id:
            QMessageBox.warning(
                self,
                "Error",
                "Cannot map detections without Instrument/Target lists.",
            )
            return

        # Try GPU first, then CPU
        device = "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            device = "cpu"

        try:
            results = self.yolo_model.predict(
                source=frame_bgr,
                conf=self.yolo_conf,
                iou=self.yolo_iou,
                device=device,
                verbose=False,
            )
        except Exception as ex:
            if device == "cuda":
                try:
                    results = self.yolo_model.predict(
                        source=frame_bgr,
                        conf=self.yolo_conf,
                        iou=self.yolo_iou,
                        device="cpu",
                        verbose=False,
                    )
                except Exception as ex2:
                    QMessageBox.warning(self, "Error", f"YOLO inference failed:\n{ex2}")
                    return
            else:
                QMessageBox.warning(self, "Error", f"YOLO inference failed:\n{ex}")
                return

        if not results:
            QMessageBox.information(self, "Info", "No detections found.")
            return

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            QMessageBox.information(self, "Info", "No detections found.")
            return

        try:
            xyxy = boxes.xyxy.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
        except Exception:
            QMessageBox.warning(self, "Error", "Failed to parse YOLO outputs.")
            return

        w = self.player._frame_w
        h = self.player._frame_h
        added = 0
        skipped = 0
        new_boxes = []
        for i in range(len(xyxy)):
            cls_id = int(cls_ids[i])
            class_name = self.class_map.get(cls_id)
            if class_name is None:
                class_name = self.class_map.get(str(cls_id))
            if class_name is None and hasattr(self.yolo_model, "names"):
                names = self.yolo_model.names
                if isinstance(names, dict):
                    class_name = names.get(cls_id)
                else:
                    try:
                        class_name = names[int(cls_id)]
                    except Exception:
                        class_name = None
            if not class_name:
                skipped += 1
                continue
            norm_name = self._norm_category(class_name)
            if norm_name in ("left_hand", "right_hand"):
                skipped += 1
                continue
            uid = category_to_default_id.get(norm_name)
            if include_ids is not None and uid not in set(include_ids or []):
                matched_uid = None
                for req_id in include_ids or []:
                    req_names = [
                        name
                        for name, val in self.global_object_map.items()
                        if val == req_id
                    ]
                    if any(self._norm_category(name) == norm_name for name in req_names):
                        matched_uid = req_id
                        break
                uid = matched_uid
            if uid is None:
                skipped += 1
                continue
            if include_ids is not None and uid not in set(include_ids or []):
                skipped += 1
                continue
            x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
            x1 = max(0.0, min(float(w), x1))
            y1 = max(0.0, min(float(h), y1))
            x2 = max(0.0, min(float(w), x2))
            y2 = max(0.0, min(float(h), y2))
            new_boxes.append(
                {
                    "id": uid,
                    "orig_frame": frame_idx - self.start_offset,
                    "label": str(class_name),
                    "class_id": cls_id,
                    "source": "yolo_detect",
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
            added += 1
        if not new_boxes and not (replace_existing and existing):
            return

        if push_undo:
            self._push_undo()
        if replace_existing and existing:
            keep = []
            for rb in self.raw_boxes:
                tgt = rb.get("orig_frame", 0) + self.start_offset
                if tgt == frame_idx and not self._is_hand_label(rb.get("label")):
                    continue
                keep.append(rb)
            self.raw_boxes = keep
        self.raw_boxes.extend(new_boxes)
        self._rebuild_bboxes_from_raw()
        self._refresh_boxes_for_frame(frame_idx)
        self._log(
            "hoi_detect_yolo",
            frame=frame_idx,
            added=added,
            skipped=skipped,
            replaced=bool(replace_existing),
        )
        # No dialog for YOLO detection; visualization and list update are enough.

    def _ensure_mp_hands(self):
        """Initialize MediaPipe Hands if available."""
        if self.mp_hands_error:
            return None
        if self.mp_hands is not None:
            return self.mp_hands
        try:
            import mediapipe as mp
        except Exception as ex:
            self.mp_hands_error = str(ex)
            QMessageBox.warning(
                self, "Missing package", f"MediaPipe is not available:\n{ex}"
            )
            return None
        if not hasattr(mp, "solutions"):
            try:
                import mediapipe.solutions as mp_solutions

                mp.solutions = mp_solutions
            except Exception as ex:
                self.mp_hands_error = str(ex)
                QMessageBox.warning(
                    self,
                    "MediaPipe Error",
                    f"MediaPipe solutions are unavailable:\n{ex}",
                )
                return None
        try:
            self.mp_hands = mp.solutions.hands.Hands(
                static_image_mode=True,
                max_num_hands=self.mp_hands_max,
                min_detection_confidence=self.mp_hands_conf,
                min_tracking_confidence=self.mp_hands_track_conf,
            )
            return self.mp_hands
        except Exception as ex:
            self.mp_hands_error = str(ex)
            QMessageBox.warning(
                self, "Error", f"Failed to initialize MediaPipe Hands:\n{ex}"
            )
            return None

    def _detect_current_frame_hands(self, frame_bgr, frame_idx=None, push_undo=True):
        """Run MediaPipe Hands on current frame and replace hand boxes for that frame."""
        mp_hands = self._ensure_mp_hands()
        if mp_hands is None:
            return 0
        if frame_bgr is None:
            return 0
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            results = mp_hands.process(rgb)
        except Exception as ex:
            if not self.mp_hands_error:
                self.mp_hands_error = str(ex)
                QMessageBox.warning(
                    self, "MediaPipe Error", f"Hand detection failed:\n{ex}"
                )
            return 0
        if not results or not results.multi_hand_landmarks:
            return 0

        handedness = []
        if results.multi_handedness:
            for hd in results.multi_handedness:
                try:
                    handedness.append(hd.classification[0].label)
                except Exception:
                    handedness.append(None)
        while len(handedness) < len(results.multi_hand_landmarks):
            handedness.append(None)

        boxes = []
        used = set()
        for idx, hand_lms in enumerate(results.multi_hand_landmarks):
            xs = [lm.x for lm in hand_lms.landmark]
            ys = [lm.y for lm in hand_lms.landmark]
            x1 = max(0.0, min(float(w), min(xs) * w))
            y1 = max(0.0, min(float(h), min(ys) * h))
            x2 = max(0.0, min(float(w), max(xs) * w))
            y2 = max(0.0, min(float(h), max(ys) * h))

            a1 = self.actors_config[0]["id"]
            a2 = self.actors_config[1]["id"] if len(self.actors_config) > 1 else a1
            
            label = None
            if idx < len(handedness) and handedness[idx]:
                h_low = str(handedness[idx]).lower()
                if h_low.startswith("left"):
                    label = self._normalize_hand_label("left")
                elif h_low.startswith("right"):
                    label = self._normalize_hand_label("right")
            if label is None:
                label = a1 if a1 not in used else a2
            if label in used:
                other = a2 if label == a1 else a1
                if other not in used:
                    label = other
                else:
                    continue
            if self.mp_hands_swap:
                label = a2 if label == a1 else a1
            if label in used:
                other = a2 if label == a1 else a1
                if other not in used:
                    label = other
                else:
                    continue
            if label in used and all(a["id"] in used for a in self.actors_config[:2]):
                continue
            used.add(label)
            boxes.append((label, x1, y1, x2, y2))

        detected_norm = {
            self._normalize_hand_label(b[0])
            for b in boxes
            if self._normalize_hand_label(b[0])
        }

        if not boxes:
            return 0

        if frame_idx is None:
            frame_idx = int(self.player.current_frame)
        if push_undo:
            self._push_undo()
        keep = []
        for rb in self.raw_boxes:
            tgt = rb.get("orig_frame", 0) + self.start_offset
            if tgt == frame_idx:
                norm = self._normalize_hand_label(rb.get("label"))
                if norm and norm in detected_norm:
                    continue
            keep.append(rb)
        self.raw_boxes = keep

        for label, x1, y1, x2, y2 in boxes:
            self.raw_boxes.append(
                {
                    "id": self.box_id_counter,
                    "orig_frame": frame_idx - self.start_offset,
                    "label": label,
                    "source": "mediapipe_hands",
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
            self.box_id_counter += 1

        self._rebuild_bboxes_from_raw()
        self._refresh_boxes_for_frame(frame_idx)
        self._log("hoi_detect_hands", frame=frame_idx, count=len(boxes))
        return len(boxes)

    def _detect_selected_action(self):
        if not self._guard_experiment_mode("detection"):
            return
        if self.selected_event_id is None:
            QMessageBox.information(
                self,
                "HOI Detection",
                "Select an action first, then use Detect Action.",
            )
            return
        self._detect_action_active_items(self.selected_event_id)

    def _detect_action_active_items(self, event_id: int):
        ev = self._find_event_by_id(event_id)
        if not ev:
            return
        keyframes = set()
        active_ids = set()
        for actor in self.actors_config:
            hand_key = actor["id"]
            hand_data = ev.get("hoi_data", {}).get(hand_key, {}) or {}
            for key in (
                "interaction_start",
                "functional_contact_onset",
                "interaction_end",
            ):
                val = hand_data.get(key)
                if val is not None:
                    keyframes.add(int(val))
            inst_id = hand_data.get("instrument_object_id")
            if inst_id is not None:
                active_ids.add(inst_id)
            target_id = hand_data.get("target_object_id")
            if target_id is not None:
                active_ids.add(target_id)

        if not keyframes:
            QMessageBox.warning(
                self,
                "HOI Detection",
                "The selected action does not have start, onset, or end frames yet.",
            )
            return

        include_ids = set(active_ids)
        if not include_ids:
            reply = QMessageBox.question(
                self,
                "HOI Detection",
                "This action has no assigned instrument or target.\nDetect all object categories on its keyframes instead?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply != QMessageBox.Yes:
                return
            include_ids = None

        orig_frame = int(self.player.current_frame)
        batch_policy = "append"
        for fidx in keyframes:
            existing = [
                b
                for b in self.bboxes.get(int(fidx), [])
                if not self._is_hand_label(b.get("label"))
            ]
            if existing:
                batch_policy = self.yolo_existing_policy or self._ask_existing_policy()
                if not batch_policy:
                    return
                break

        self._push_undo()
        detected_frames = 0
        try:
            for fidx in sorted(keyframes):
                self.player.seek(int(fidx))
                self._detect_current_frame_yolo(
                    frame_idx=int(fidx),
                    include_ids=include_ids,
                    override_policy=batch_policy,
                    push_undo=False,
                )
                detected_frames += 1
        finally:
            self.player.seek(orig_frame)
            self._refresh_boxes_for_frame(orig_frame)
            self._update_overlay(orig_frame)
        self._log(
            "hoi_detect_action_keyframes",
            event_id=event_id,
            frames=detected_frames,
            filtered=bool(include_ids),
        )

    def _detect_all_actions(self):
        if not self._guard_experiment_mode("detection"):
            return
        if not self.events:
            QMessageBox.information(self, "HOI Detection", "No actions are available.")
            return

        frame_to_active_ids = {}
        for ev in self.events:
            for actor in self.actors_config:
                hand_key = actor["id"]
                hand_data = ev.get("hoi_data", {}).get(hand_key, {}) or {}
                keyframes = set()
                for key in (
                    "interaction_start",
                    "functional_contact_onset",
                    "interaction_end",
                ):
                    val = hand_data.get(key)
                    if val is not None:
                        keyframes.add(int(val))
                if not keyframes:
                    continue
                active_ids = set()
                inst_id = hand_data.get("instrument_object_id")
                if inst_id is not None:
                    active_ids.add(inst_id)
                target_id = hand_data.get("target_object_id")
                if target_id is not None:
                    active_ids.add(target_id)
                for fidx in keyframes:
                    frame_to_active_ids.setdefault(int(fidx), set()).update(active_ids)

        if not frame_to_active_ids:
            QMessageBox.information(
                self,
                "HOI Detection",
                "No action keyframes with assigned instrument or target were found.",
            )
            return

        orig_frame = int(self.player.current_frame)
        batch_policy = "append"
        for fidx in frame_to_active_ids.keys():
            existing = [
                b
                for b in self.bboxes.get(int(fidx), [])
                if not self._is_hand_label(b.get("label"))
            ]
            if existing:
                batch_policy = self.yolo_existing_policy or self._ask_existing_policy()
                if not batch_policy:
                    return
                break

        self._push_undo()
        detected_frames = 0
        try:
            for fidx, active_ids in sorted(frame_to_active_ids.items()):
                self.player.seek(int(fidx))
                self._detect_current_frame_yolo(
                    frame_idx=int(fidx),
                    include_ids=active_ids if active_ids else None,
                    override_policy=batch_policy,
                    push_undo=False,
                )
                detected_frames += 1
        finally:
            self.player.seek(orig_frame)
            self._refresh_boxes_for_frame(orig_frame)
            self._update_overlay(orig_frame)
        self._log("hoi_detect_all_action_keyframes", frames=detected_frames)

    def _incremental_train_yolo(self):
        """Main entry point for YOLO fine-tuning on current annotations."""
        if not self._guard_experiment_mode("detection"):
            return
        if not self.yolo_weights_path or not self.video_path:
            QMessageBox.warning(self, "Error", "Please load both a video and a YOLO model first.")
            return

        if not self.class_map:
             QMessageBox.warning(self, "Error", "Please load a class map (data.yaml) first.")
             return

        # 1. Ask for config
        dlg = YoloTrainDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        config = dlg.get_config()

        # 2. Prepare dataset
        try:
            yaml_path = self._prepare_yolo_dataset(config)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to prepare dataset:\n{e}")
            return

        # 3. Start training
        self._log("hoi_yolo_train_start", weights=self.yolo_weights_path, config=config)
        self.train_worker = YoloTrainWorker(self.yolo_weights_path, yaml_path, config)
        self.train_worker.progress.connect(self._on_train_progress)
        self.train_worker.finished.connect(self._on_train_finished)
        
        self.train_progress_dlg = QProgressDialog("Initializing training...", "Cancel", 0, 0, self)
        self.train_progress_dlg.setWindowTitle("YOLO Training")
        self.train_progress_dlg.setWindowModality(Qt.WindowModal)
        self.train_progress_dlg.setMinimumDuration(0)
        self.train_progress_dlg.canceled.connect(self.train_worker.terminate) # Simple kill if canceled
        self.train_progress_dlg.show()
        
        self.train_worker.start()

    def _prepare_yolo_dataset(self, config):
        """Harvest frames and boxes from raw_boxes to create a YOLO dataset."""
        import shutil
        import random
        
        base_dir = config["output_dir"]
        os.makedirs(base_dir, exist_ok=True)
        
        # YOLO structure
        img_train = os.path.join(base_dir, "images", "train")
        img_val = os.path.join(base_dir, "images", "val")
        lbl_train = os.path.join(base_dir, "labels", "train")
        lbl_val = os.path.join(base_dir, "labels", "val")
        
        for d in [img_train, img_val, lbl_train, lbl_val]:
            if os.path.exists(d):
                try:
                    shutil.rmtree(d)
                except Exception:
                    pass
            os.makedirs(d, exist_ok=True)

        # Reverse class map: normalized_name -> cid
        name_to_cid = {self._norm_category(v): k for k, v in self.class_map.items()}
        
        # Collect frames to export (matching annotated objects with class_map)
        frame_to_boxes = {}
        for b in self.raw_boxes:
            if self._is_hand_label(b.get("label")):
                continue
            
            lbl = self._norm_category(b.get("label", ""))
            if lbl not in name_to_cid:
                continue
            
            f = b.get("orig_frame")
            if f not in frame_to_boxes:
                frame_to_boxes[f] = []
            frame_to_boxes[f].append(b)
        
        if not frame_to_boxes:
            raise Exception("No annotated objects found in your library that match classes in data.yaml.\n"
                            "Check if names match exactly (ignoring case/underscores).")

        # Export frames
        cap = cv2.VideoCapture(self.video_path)
        frame_indices = list(frame_to_boxes.keys())
        random.shuffle(frame_indices)
        
        split_idx = int(len(frame_indices) * config["train_split"])
        train_frames = frame_indices[:split_idx]
        
        for fidx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret: continue
            
            h, w = frame.shape[:2]
            subset = "train" if fidx in train_frames else "val"
            
            img_filename = f"frame_{fidx}.jpg"
            img_path = os.path.join(base_dir, "images", subset, img_filename)
            lbl_path = os.path.join(base_dir, "labels", subset, f"frame_{fidx}.txt")
            
            cv2.imwrite(img_path, frame)
            
            with open(lbl_path, "w") as f_lbl:
                for b in frame_to_boxes[fidx]:
                    lbl = self._norm_category(b.get("label", ""))
                    cid = name_to_cid[lbl]
                    # Normalized YOLO: xc yc w h
                    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
                    bw = (bx2 - bx1) / w
                    bh = (by2 - by1) / h
                    bxc = (bx1 + bx2) / (2 * w)
                    byc = (by1 + by2) / (2 * h)
                    f_lbl.write(f"{cid} {bxc:.6f} {byc:.6f} {bw:.6f} {bh:.6f}\n")
        
        cap.release()
        
        # Generate dataset.yaml
        yaml_content = {
            "path": os.path.abspath(base_dir),
            "train": "images/train",
            "val": "images/val",
            "names": {int(k): v for k, v in self.class_map.items()}
        }
        yaml_path = os.path.join(base_dir, "dataset.yaml")
        with open(yaml_path, "w") as f_yaml:
            yaml.dump(yaml_content, f_yaml, sort_keys=False)
            
        return yaml_path

    def _on_train_progress(self, msg):
        if hasattr(self, "train_progress_dlg") and self.train_progress_dlg:
            self.train_progress_dlg.setLabelText(msg)

    def _on_train_finished(self, success, message):
        if hasattr(self, "train_progress_dlg") and self.train_progress_dlg:
            self.train_progress_dlg.close()
            self.train_progress_dlg.deleteLater()
            self.train_progress_dlg = None
        
        # Give a small buffer for thread to fully unwind
        QTimer.singleShot(100, lambda: self._show_train_result(success, message))

    def _show_train_result(self, success, message):
        if success:
            self._log("hoi_yolo_train_finished", success=True)
            QMessageBox.information(self, "Training Finished", message)
        else:
            self._log("hoi_yolo_train_finished", success=False, error=message)
            QMessageBox.critical(self, "Training Error", f"Training failed:\n{message}")

    def _detect_current_frame_combined(self):
        """Run YOLO object detection and MediaPipe hand detection on the current frame."""
        if not self._guard_experiment_mode("detection"):
            return
        if not self.player.cap:
            QMessageBox.warning(self, "Missing prerequisite", "Please load a video first.")
            return
        frame_bgr = self.player.get_current_frame_bgr()
        if frame_bgr is None:
            QMessageBox.warning(self, "Error", "No video frame available.")
            return

        # Objects (YOLO)
        if self.class_map and self.global_object_map:
            self._detect_current_frame_yolo()
        else:
            QMessageBox.information(
                self,
                "Info",
                "Object detection skipped. Load a class map and Instrument/Target lists first.",
            )

        # Hands (MediaPipe)
        self._detect_current_frame_hands(frame_bgr)

    def _save_annotations_json(self):
        """Save HOI annotations with optional integrity checks."""
        try:
            # persist any in-progress edits before validation/export
            if self.selected_event_id is not None:
                self._save_ui_to_hand_draft(self.selected_hand_label)
                self._apply_draft_to_selected_event()

            # 1. Basic check
            if not self.events and not self.raw_boxes:
                QMessageBox.information(self, "Info", "No HOI data to save.")
                return

            # 2. Generate filename
            base = os.path.splitext(os.path.basename(self.video_path or "annotation"))[
                0
            ]
            default_name = f"{base}_hoi.json"
            fp, _ = QFileDialog.getSaveFileName(
                self,
                "Save HOI Annotations",
                default_name,
                "JSON Files (*.json);;All Files (*)",
            )
            if not fp:
                return

            validation_error = None

            for i, event in enumerate(self.events):
                event_id = event["event_id"]
                for actor in self.actors_config:
                    hand_key = actor["id"]
                    h_data = event["hoi_data"][hand_key]

                    has_intent = (
                        bool(h_data.get("verb"))
                        or h_data.get("interaction_start") is not None
                    )

                    if has_intent:
                        errs = self._validate_integrity(h_data)
                        if errs:
                            err = errs[0]
                            validation_error = (
                                f"Event {event_id} ({hand_key}):\n{err['msg']}"
                            )
                            break
                if validation_error:
                    break

            ok_incomplete, _ = self._check_incomplete_hoi(context="save")
            if not ok_incomplete:
                return

            # -----------------------------------------------------------
            # Toggle Block
            # -----------------------------------------------------------
            # if validation_error:
            #     reply = QMessageBox.question(
            #         self, "Integrity Warning",
            #         f"{validation_error}\n\n(Object bbox might be missing at this frame)\nForce save anyway?",
            #         QMessageBox.Yes | QMessageBox.No
            #     )
            #     if reply == QMessageBox.No:
            #         self.player.seek(error_frame)
            #         self._refresh_boxes_for_frame(error_frame)
            #         return
            # -----------------------------------------------------------

            # 4. Build payload
            payload = self._build_payload_v2()
            self._refresh_sparse_evidence_snapshots()

            # 5. Write file
            with open(fp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)

            try:
                graph = build_hoi_event_graph(
                    self.events,
                    video_path=self.video_path,
                    annotation_path=fp,
                    actors_config=self.actors_config,
                )
                graph_path = save_event_graph_sidecar(fp, graph)
                self._log(
                    "hoi_save_event_graph",
                    path=graph_path,
                    anchors=len(graph.get("onset_anchors", []) or []),
                    locked_regions=len(graph.get("locked_regions", []) or []),
                    consistency_issues=len(graph.get("consistency_flags", []) or []),
                    sparse_evidence_expected=(
                        (graph.get("stats", {}) or {}).get(
                            "sparse_evidence_expected_count", 0
                        )
                    ),
                    sparse_evidence_missing=(
                        (graph.get("stats", {}) or {}).get(
                            "sparse_evidence_missing_count", 0
                        )
                    ),
                )
            except Exception as graph_ex:
                self._log(
                    "hoi_save_event_graph_failed",
                    path=fp,
                    error=str(graph_ex),
                )

            self._log(
                "hoi_save_annotations",
                path=fp,
                events=len(self.events),
                boxes=len(self.raw_boxes),
            )
            log_base = os.path.splitext(fp)[0]
            if log_base:
                self._flush_ops_log_safely(
                    getattr(self, "op_logger", None),
                    log_base + ".ops.log.csv",
                    context="HOI save",
                )
            if (
                log_base
                and getattr(self, "validation_op_logger", None)
                and self.validation_op_logger.enabled
                and self.validation_session_active
            ):
                self._flush_ops_log_safely(
                    self.validation_op_logger,
                    log_base + ".validation.ops.log.csv",
                    context="HOI validation save",
                )
            self._save_validation_summary(fp, payload)
            self._mark_hoi_saved()
            QMessageBox.information(
                self, "Saved", f"Successfully saved to:\n{os.path.basename(fp)}"
            )

        except Exception as ex:
            import traceback

            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Save failed:\n{str(ex)}")

    def _collect_incomplete_hoi(self) -> List[dict]:
        issues = []
        for ev in self.events:
            ev_id = ev.get("event_id")
            for actor in self.actors_config:
                hand_key = actor["id"]
                h_data = ev.get("hoi_data", {}).get(hand_key, {}) or {}
                start = h_data.get("interaction_start")
                end = h_data.get("interaction_end")
                onset = h_data.get("functional_contact_onset")
                has_segment = start is not None or end is not None or onset is not None
                has_meta = (
                    bool(h_data.get("verb"))
                    or h_data.get("instrument_object_id") is not None
                    or h_data.get("target_object_id") is not None
                )
                if not (has_segment or has_meta):
                    continue
                labels = self._parse_anomaly_labels(h_data.get("anomaly_label"))
                allow_missing_bbox = self._anomaly_rule_allows(
                    labels, "allow_missing_bbox"
                )
                allow_missing_verb = self._anomaly_rule_allows(
                    labels, "allow_missing_verb"
                )
                missing = []
                if start is None or end is None:
                    missing.append("start/end")
                if onset is None:
                    missing.append("onset")
                if not h_data.get("verb") and not allow_missing_verb:
                    missing.append("verb")
                if h_data.get("instrument_object_id") is None:
                    missing.append("instrument")
                if h_data.get("target_object_id") is None:
                    missing.append("target")
                if not allow_missing_bbox:
                    bbox_errors = self._validate_integrity(h_data)
                    if bbox_errors:
                        time_priority = {"onset": 0, "start": 1, "end": 2}
                        bbox_errors = sorted(
                            bbox_errors,
                            key=lambda err: (
                                time_priority.get(
                                    str(err.get("time_key") or "").strip().lower(), 3
                                ),
                                str(err.get("role_key") or "").strip().lower(),
                            ),
                        )
                        preview_slots = []
                        for err in bbox_errors[:2]:
                            role_key = str(err.get("role_key") or "").strip().lower()
                            time_key = str(err.get("time_key") or "").strip().lower()
                            if role_key and time_key:
                                preview_slots.append(f"{role_key}@{time_key}")
                        if preview_slots:
                            label = "bbox " + ", ".join(preview_slots)
                            if len(bbox_errors) > len(preview_slots):
                                label += f" (+{len(bbox_errors) - len(preview_slots)})"
                            missing.append(label)
                        else:
                            missing.append(f"bbox ({len(bbox_errors)})")
                if missing:
                    frame = (
                        onset
                        if onset is not None
                        else (
                            start
                            if start is not None
                            else end if end is not None else None
                        )
                    )
                    issues.append(
                        {
                            "event_id": ev_id,
                            "hand": hand_key,
                            "missing": missing,
                            "frame": frame,
                        }
                    )
        return issues

    def _check_incomplete_hoi(self, context: str = "save") -> tuple:
        issues = self._collect_incomplete_hoi()
        if not issues:
            return True, False
        preview = []
        for entry in issues[:4]:
            hand_short = self._get_actor_short_label(entry["hand"])
            missing = ", ".join(entry["missing"])
            preview.append(f"Event {entry['event_id']} {hand_short}: {missing}")
        if len(issues) > 4:
            preview.append(f"... and {len(issues) - 4} more")
        msg = (
            "Incomplete HandOI segments detected.\n"
            "Missing fields: start/end, onset, verb, instrument, target, or bbox.\n\n"
            + "\n".join(preview)
            + "\n\nContinue anyway?"
        )
        reply = QMessageBox.question(
            self, "Incomplete annotations", msg, QMessageBox.Yes | QMessageBox.No
        )
        proceed = reply == QMessageBox.Yes
        self._log(
            "hoi_incomplete_check",
            context=context,
            has_issues=True,
            proceed=proceed,
            count=len(issues),
        )
        return proceed, True

    def _update_incomplete_indicator(self):
        issues = self._collect_incomplete_hoi()
        self._incomplete_issues = issues
        if not issues:
            self.lbl_incomplete.setText("Incomplete: none")
            self.lbl_incomplete.setToolTip("No incomplete HandOI segments detected.")
            self._set_status_chip(getattr(self, "lbl_incomplete_chip", None), "Incomplete 0", "ok")
            if getattr(self, "lbl_review_status", None):
                self.lbl_review_status.setText("No incomplete HandOI segments detected.")
                self.lbl_review_status.setToolTip("No incomplete HandOI segments detected.")
            self._set_status_card_tone(
                getattr(self, "review_status_card", None),
                "active" if bool(getattr(self, "validation_enabled", False)) else "ok",
            )
            self.btn_incomplete_prev.setEnabled(False)
            self.btn_incomplete_next.setEnabled(False)
            if getattr(self, "btn_review_prev", None):
                self.btn_review_prev.setEnabled(False)
            if getattr(self, "btn_review_next", None):
                self.btn_review_next.setEnabled(False)
            self._incomplete_idx = -1
            self._update_inspector_tab_labels()
            return
        self._incomplete_idx = min(
            self._incomplete_idx if self._incomplete_idx >= 0 else 0, len(issues) - 1
        )
        self.lbl_incomplete.setText(f"Incomplete: {len(issues)}")
        tooltip = []
        for entry in issues[:6]:
            hand = self._get_actor_short_label(entry["hand"])
            missing = ", ".join(entry["missing"])
            tooltip.append(f"Event {entry['event_id']} {hand}: {missing}")
        if len(issues) > 6:
            tooltip.append("...")
        tooltip_text = "\n".join(tooltip)
        self.lbl_incomplete.setToolTip(tooltip_text)
        self._set_status_chip(getattr(self, "lbl_incomplete_chip", None), f"Incomplete {len(issues)}", "warn")
        if getattr(self, "lbl_review_status", None):
            self.lbl_review_status.setText(tooltip_text)
            self.lbl_review_status.setToolTip(tooltip_text)
        self._set_status_card_tone(getattr(self, "review_status_card", None), "warn")
        self.btn_incomplete_prev.setEnabled(True)
        self.btn_incomplete_next.setEnabled(True)
        if getattr(self, "btn_review_prev", None):
            self.btn_review_prev.setEnabled(True)
        if getattr(self, "btn_review_next", None):
            self.btn_review_next.setEnabled(True)
        self._update_inspector_tab_labels()

    def _jump_incomplete(self, direction: int):
        if not self._incomplete_issues:
            return
        self._incomplete_idx = (self._incomplete_idx + direction) % len(
            self._incomplete_issues
        )
        issue = self._incomplete_issues[self._incomplete_idx]
        frame = issue.get("frame")
        if frame is None:
            frame = int(getattr(self.player, "current_frame", 0))
        if self.player.cap:
            self.player.seek(frame)
        self._refresh_boxes_for_frame(frame)
        self._set_frame_controls(frame)
        event_id = issue.get("event_id")
        hand = issue.get("hand")
        if event_id is not None and self._is_hand_label(hand):
            self._set_selected_event(event_id, hand)
        self._focus_inspector_tab("review")
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.set_current_frame(frame)
            self.hoi_timeline.refresh()
        self._log(
            "hoi_incomplete_jump",
            direction=direction,
            target=frame,
            event_id=event_id,
            hand=hand,
        )

    def _save_validation_summary(self, annotations_path: str, payload: dict):
        if not bool(getattr(self, "validation_summary_enabled", True)):
            return
        if not self.validation_session_active:
            return
        summary = {
            "version": "HOI_VALIDATION_V1",
            "editor": self.validator_name,
            "validation_started_at": self.validation_started_at,
            "validation_saved_at": datetime.now().isoformat(timespec="seconds"),
            "video_path": self.video_path,
            "annotation_path": annotations_path,
            "events": len(self.events),
            "boxes": len(self.raw_boxes),
            "modified": bool(self.validation_modified),
            "change_count": int(self.validation_change_count),
            "final_state": payload,
        }
        log_path = os.path.splitext(annotations_path)[0] + ".validation.json"
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=True)
            print(f"[LOG] {log_path}")
            self._log("hoi_validation_summary_saved", path=log_path)
        except Exception as ex:
            print(f"[LOG][ERROR] HOI validation summary write failed: {log_path} ({ex})")
            QMessageBox.warning(
                self,
                "Validation log",
                f"Failed to write validation log:\n{log_path}\n\n{ex}",
            )

    def _init_anomaly_rules(self):
        self.anomaly_rules = {}
        rules_dict = self.extra_label_config.get("rules", {})
        for label in self.anomaly_labels:
            config_rule = rules_dict.get(label, {})
            self.anomaly_rules[label] = {
                "allow_missing_bbox": config_rule.get("allow_missing_bbox", False),
                "allow_missing_verb": config_rule.get("allow_missing_verb", False),
            }

    def _ensure_anomaly_rules(self):
        for label in self.anomaly_labels:
            if label not in self.anomaly_rules:
                self.anomaly_rules[label] = {
                    "allow_missing_bbox": False,
                    "allow_missing_verb": False,
                }
        for label in list(self.anomaly_rules.keys()):
            if label not in self.anomaly_labels:
                del self.anomaly_rules[label]

    def _normalize_anomaly_label(self, label: str) -> str:
        if not label:
            return self.extra_label_config.get("default_label", "Normal")
        text = str(label).strip()
        if text.lower() == "correct":
            return self.extra_label_config.get("default_label", "Normal")
        return text

    def _parse_anomaly_labels(self, label_str: str) -> List[str]:
        if not label_str:
            return [self.extra_label_config.get("default_label", "Normal")]
        raw = [
            self._normalize_anomaly_label(t.strip())
            for t in str(label_str).split(",")
            if t.strip()
        ]
        if not raw:
            return [self.extra_label_config.get("default_label", "Normal")]
        default_lbl = self.extra_label_config.get("default_label", "Normal")
        if default_lbl in raw and len(raw) > 1:
            raw = [t for t in raw if t != fallback]
        # preserve order, de-dupe
        seen = set()
        out = []
        for t in raw:
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
        return out

    def _anomaly_rule_allows(self, labels: List[str], key: str) -> bool:
        self._ensure_anomaly_rules()
        for label in labels:
            for rule_key, rule in self.anomaly_rules.items():
                if rule_key.strip().lower() == label.strip().lower():
                    if rule.get(key):
                        return True
        return False

    def _edit_anomaly_rules(self):
        self._ensure_anomaly_rules()
        dialog = QDialog(self)
        dialog.setWindowTitle("Anomaly Rules")
        dialog.resize(480, 320)
        layout = QVBoxLayout(dialog)
        table = QTableWidget(dialog)
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(
            ["Anomaly label", "Allow missing bbox", "Allow missing verb"]
        )
        table.verticalHeader().setVisible(False)
        table.setRowCount(len(self.anomaly_labels))
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        for row_idx, label in enumerate(self.anomaly_labels):
            name_item = QTableWidgetItem(label)
            table.setItem(row_idx, 0, name_item)
            rule = self.anomaly_rules.get(label, {})
            bbox_item = QTableWidgetItem("")
            bbox_item.setFlags(
                bbox_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled
            )
            bbox_item.setCheckState(
                Qt.Checked if rule.get("allow_missing_bbox") else Qt.Unchecked
            )
            table.setItem(row_idx, 1, bbox_item)
            verb_item = QTableWidgetItem("")
            verb_item.setFlags(
                verb_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled
            )
            verb_item.setCheckState(
                Qt.Checked if rule.get("allow_missing_verb") else Qt.Unchecked
            )
            table.setItem(row_idx, 2, verb_item)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        layout.addWidget(table)
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dialog
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return
        for row_idx, label in enumerate(self.anomaly_labels):
            bbox_item = table.item(row_idx, 1)
            verb_item = table.item(row_idx, 2)
            self.anomaly_rules[label] = {
                "allow_missing_bbox": (
                    bbox_item.checkState() == Qt.Checked if bbox_item else False
                ),
                "allow_missing_verb": (
                    verb_item.checkState() == Qt.Checked if verb_item else False
                ),
            }
        self._log("hoi_anomaly_rules_update", rules=self.anomaly_rules)

    def _load_annotations_json(self):
        """
        Load HOI annotations.
        Supports unified per-hand event format and legacy window format.
        """
        fp, _ = QFileDialog.getOpenFileName(
            self, "Load HOI annotations", "", "JSON Files (*.json);;All Files (*)"
        )
        if not fp:
            return
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as ex:
            QMessageBox.warning(self, "Error", f"Failed to read JSON:\n{ex}")
            return
        if not isinstance(data, dict):
            QMessageBox.warning(self, "Error", "Invalid JSON: root must be an object.")
            return
        self._clear_undo_history()

        if isinstance(data, dict) and "tracks" in data and "hoi_events" in data:
            self._load_annotations_v2(data)
            self._log(
                "hoi_load_annotations",
                path=fp,
                events=len(self.events),
                boxes=len(self.raw_boxes),
            )
            self._mark_hoi_saved()
            return

        # --- 1. Restore Meta Data (Object Lookup) ---
        meta = data.get("meta_data", {})
        if not isinstance(meta, dict):
            meta = {}
        obj_lookup = meta.get("object_lookup", {})
        if not isinstance(obj_lookup, dict):
            obj_lookup = {}

        self.global_object_map.clear()
        self.object_id_counter = 0
        self.combo_instrument.clear()
        self.combo_target.clear()
        self.combo_instrument.addItem("None", None)
        self.combo_target.addItem("None", None)

        # Restore objects by ID
        sorted_ids = []
        for k in obj_lookup.keys():
            try:
                sorted_ids.append(int(k))
            except Exception:
                continue
        sorted_ids.sort()
        for uid in sorted_ids:
            info = obj_lookup.get(str(uid), {})
            name = info.get("name", f"obj_{uid}")
            category = info.get("category", "unknown")

            self.global_object_map[name] = uid
            self.id_to_category[name] = category

            if uid >= self.object_id_counter:
                self.object_id_counter = uid + 1

            display_text = f"[{uid}] {name}"
            self.combo_instrument.addItem(display_text, uid)
            self.combo_target.addItem(display_text, uid)

        # --- 2. Restore Action Lookup ---
        action_lookup = meta.get("action_lookup", {})
        if isinstance(action_lookup, dict) and action_lookup:
            self.verbs.clear()
            sorted_vids = []
            for k in action_lookup.keys():
                try:
                    sorted_vids.append(int(k))
                except Exception:
                    continue
            sorted_vids.sort()
            for vid in sorted_vids:
                vname = action_lookup[str(vid)]
                color = self._color_for_index(vid)
                self.verbs.append(LabelDef(name=vname, id=vid, color_name=color))

            self._renumber_verbs()
            self._update_verb_combo()
            self.label_panel.refresh()

        # --- 3. Restore Events and BBoxes ---
        self.events.clear()
        self.event_id_counter = 0

        # Map for deduplication of restored boxes
        restored_boxes_map = {}

        loaded_windows = data.get("windows", [])

        for w_item in loaded_windows:
            # A. Basic Event Info (legacy window_id)
            event_id = w_item.get("window_id", self.event_id_counter)
            frames = w_item.get("frames", [0, 0])
            anomalies = w_item.get("anomaly_labels", {})
            anomaly_label = self.extra_label_config.get("default_label", "Normal")
            if isinstance(anomalies, dict):
                for k, v in anomalies.items():
                    try:
                        if int(v) == 1:
                            anomaly_label = self._normalize_anomaly_label(str(k))
                            break
                    except Exception:
                        continue

            if event_id >= self.event_id_counter:
                self.event_id_counter = event_id + 1

            hoi_data = {
                actor["id"]: {
                    "verb": "",
                    "instrument_object_id": None,
                    "target_object_id": None,
                    "interaction_start": None,
                    "functional_contact_onset": None,
                    "interaction_end": None,
                    "anomaly_label": anomaly_label,
                }
                for actor in self.actors_config
            }

            # B. Parse Hands
            hands_list = w_item.get("hands", [])
            for h in hands_list:
                hid_str = h.get("hand_id", "").lower()

                internal_key = None
                box_label = None

                if "left" in hid_str:
                    internal_key = self.actors_config[0]["id"]
                    box_label = internal_key
                elif "right" in hid_str:
                    internal_key = self.actors_config[1]["id"] if len(self.actors_config) > 1 else self.actors_config[0]["id"]
                    box_label = internal_key

                if internal_key:
                    hoi_entry = h.get("hoi", {})
                    hoi_data[internal_key].update(
                        {
                            "verb": hoi_entry.get("verb", ""),
                            "instrument_object_id": hoi_entry.get(
                                "instrument_object_id"
                            ),
                            "target_object_id": hoi_entry.get("target_object_id"),
                            "interaction_start": hoi_entry.get("interaction_start"),
                            "functional_contact_onset": hoi_entry.get(
                                "functional_contact_onset"
                            ),
                            "interaction_end": hoi_entry.get("interaction_end"),
                        }
                    )

                # Restore Hand BBox Track
                if box_label:
                    tracks = h.get("bbox_track", {})
                    for f_str, coords in tracks.items():
                        f_idx = int(f_str)
                        key = (f_idx, box_label)
                        if key not in restored_boxes_map:
                            restored_boxes_map[key] = {
                                "id": self.box_id_counter,
                                "orig_frame": f_idx,
                                "label": box_label,
                                "x1": coords[0],
                                "y1": coords[1],
                                "x2": coords[2],
                                "y2": coords[3],
                            }
                            self.box_id_counter += 1

            # C. Parse Objects
            objs_list = w_item.get("objects", [])
            for obj in objs_list:
                uid = obj.get("object_id")
                obj_name = ""
                for k, v in self.global_object_map.items():
                    if v == uid:
                        obj_name = k
                        break

                tracks = obj.get("bbox_track", {})
                for f_str, coords in tracks.items():
                    f_idx = int(f_str)
                    key = (f_idx, uid)
                    if key not in restored_boxes_map:
                        restored_boxes_map[key] = {
                            "id": uid,
                            "orig_frame": f_idx,
                            "label": obj_name,
                            "x1": coords[0],
                            "y1": coords[1],
                            "x2": coords[2],
                            "y2": coords[3],
                        }

            # D. Build Event object
            new_event = {"event_id": event_id, "frames": frames, "hoi_data": hoi_data}
            self.events.append(new_event)

        # --- 4. Write restored BBoxes to memory ---
        if restored_boxes_map:
            self.raw_boxes.extend(restored_boxes_map.values())
            self._rebuild_bboxes_from_raw()
            self._refresh_boxes_for_frame(self.player.current_frame)

        # --- 5. Refresh UI ---
        self._ensure_verbs_cover_events()
        self._refresh_events()
        self._after_events_loaded()

        video_name = meta.get("video_name", "unknown")
        box_count = len(restored_boxes_map)
        QMessageBox.information(
            self,
            "Loaded",
            f"Successfully loaded {len(self.events)} events.\n"
            f"Restored {box_count} keyframe BBoxes.\n"
            f"Video source: {video_name}",
        )
        self._log(
            "hoi_load_annotations_legacy",
            path=fp,
            events=len(self.events),
            boxes=box_count,
        )
        self._mark_hoi_saved()

    def _load_annotations_v2(self, data: dict):
        """Load unified per-hand event format."""
        self._clear_undo_history()
        # Reset state
        self.events.clear()
        self.event_id_counter = 0
        self.raw_boxes = []
        self.bboxes = {}
        self._clear_relation_highlight()
        self.video_path = data.get("video_path", "") or self.video_path

        # Restore Extra Configs
        if "extra_label_config" in data:
            self.extra_label_config = data["extra_label_config"]
            self._apply_extra_label_config()

        if "actors_config" in data:
            self.actors_config = data["actors_config"]
            # Rebuild UI if needed? For now just keep it
        else:
            # Revert to default Hand Anomaly config if missing
            self.extra_label_config = {
                "title": "Hand Anomaly Label",
                "default_label": "Normal",
                "labels": [
                    "Visibility / Occlusion",
                    "Handling Anomaly",
                    "Tool Anomaly",
                    "Part Mismatch",
                    "Spatial Anomaly",
                    "Temporal Anomaly",
                    "Quality / Outcome-visible Defect",
                    "Completion / Finish Missing",
                    "Detach / Fall-out",
                    "Recovery / Rework",
                    "Normal",
                ],
                "rules": {
                    "Visibility / Occlusion": {
                        "allow_missing_bbox": True,
                        "allow_missing_verb": True,
                    }
                },
            }
            self._apply_extra_label_config()

        # Restore libraries
        self.global_object_map.clear()
        self.id_to_category.clear()
        self.object_id_counter = 0
        self.combo_instrument.clear()
        self.combo_target.clear()
        self.combo_instrument.addItem("None", None)
        self.combo_target.addItem("None", None)

        obj_lib = data.get("object_library", {})
        obj_class_map = {}
        max_obj_id = -1
        for id_str in sorted(obj_lib.keys(), key=lambda x: int(x)):
            try:
                uid = int(id_str)
            except Exception:
                continue
            info = obj_lib.get(id_str, {}) or {}
            label = (
                info.get("label")
                or info.get("name")
                or info.get("category")
                or f"obj_{uid}"
            )
            class_id = info.get("class_id")
            if class_id is not None:
                try:
                    class_id = int(class_id)
                except Exception:
                    pass
                obj_class_map[uid] = class_id
                if (
                    label
                    and class_id not in self.class_map
                    and str(class_id) not in self.class_map
                ):
                    self.class_map[class_id] = label
            self.global_object_map[label] = uid
            self.id_to_category[label] = label
            max_obj_id = max(max_obj_id, uid)
            display_text = f"[{uid}] {label}"
            self.combo_instrument.addItem(display_text, uid)
            self.combo_target.addItem(display_text, uid)
        self.object_id_counter = max_obj_id + 1
        self.box_id_counter = max(self.object_id_counter + 1, 1)

        if "verb_library" in data:
            verb_lib = data.get("verb_library", {}) or {}
            self.verbs.clear()
            for vid_str in sorted(verb_lib.keys(), key=lambda x: int(x)):
                try:
                    vid = int(vid_str)
                except Exception:
                    continue
                name = verb_lib.get(vid_str, "")
                color = self._color_for_index(vid)
                self.verbs.append(LabelDef(name=name, id=vid, color_name=color))
            self._renumber_verbs()
            self._update_verb_combo()
            self.label_panel.refresh()
        rules = data.get("anomaly_rules", {})
        if isinstance(rules, dict) and rules:
            self.anomaly_rules = {}
            for label, rule in rules.items():
                if not isinstance(rule, dict):
                    continue
                self.anomaly_rules[str(label)] = {
                    "allow_missing_bbox": bool(rule.get("allow_missing_bbox")),
                    "allow_missing_verb": bool(rule.get("allow_missing_verb")),
                }
        self._ensure_anomaly_rules()

        # Tracks -> raw boxes
        tracks = data.get("tracks", {})
        track_obj_id = {}

        def _label_for_id(uid):
            for name, id_val in self.global_object_map.items():
                if id_val == uid:
                    return name
            return None

        for track_id, info in tracks.items():
            info = info or {}
            category = str(info.get("category", "")).strip()
            obj_id = info.get("object_id", None)
            track_class_id = info.get("class_id")
            if track_class_id is None and obj_id is not None:
                track_class_id = obj_class_map.get(obj_id)
            track_obj_id[track_id] = obj_id
            boxes = info.get("boxes", []) or []

            tid_lower = str(track_id).lower()
            cat_lower = category.lower()

            label = None
            for actor in self.actors_config:
                aid = actor["id"]
                aid_lower = aid.lower()
                if (cat_lower == aid_lower) or (aid_lower in tid_lower):
                    label = aid
                    break

            if not label:
                label = _label_for_id(obj_id) or category or f"obj_{obj_id}"

            is_actor = any(actor["id"] == label for actor in self.actors_config)

            for entry in boxes:
                frame = entry.get("frame", None)
                bbox = entry.get("bbox", None)
                if frame is None or not bbox or len(bbox) != 4:
                    continue
                try:
                    f_idx = int(frame)
                    x1, y1, x2, y2 = [float(v) for v in bbox]
                except Exception:
                    continue
                box_class_id = entry.get("class_id", track_class_id)
                if box_class_id is not None:
                    try:
                        box_class_id = int(box_class_id)
                    except Exception:
                        pass
                if is_actor or obj_id is None:
                    bid = self.box_id_counter
                    self.box_id_counter += 1
                else:
                    bid = obj_id
                if not is_actor and box_class_id is not None and label:
                    if (
                        box_class_id not in self.class_map
                        and str(box_class_id) not in self.class_map
                    ):
                        self.class_map[box_class_id] = label
                new_rb = {
                    "id": bid,
                    "orig_frame": f_idx - self.start_offset,
                    "label": label,
                    "source": "loaded_annotation",
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
                if not is_actor and box_class_id is not None:
                    new_rb["class_id"] = box_class_id
                self.raw_boxes.append(new_rb)

        self._rebuild_bboxes_from_raw()
        self._refresh_boxes_for_frame(self.player.current_frame)

        # Events -> internal event entries
        events = data.get("hoi_events", {})

        def _event_to_entry(side_key: str, event: dict):
            s = event.get("start_frame")
            o = event.get("contact_onset_frame")
            e = event.get("end_frame")
            links = event.get("links", {}) or {}
            verb = event.get("verb") or ""
            anomaly = self._normalize_anomaly_label(event.get("anomaly_label"))
            target_tid = links.get("target_track_id")
            tool_tid = links.get("tool_track_id")
            target_id = track_obj_id.get(target_tid) if target_tid else None
            tool_id = track_obj_id.get(tool_tid) if tool_tid else None
            annotation_state = event.get("annotation_state", {}) or {}

            def _safe_int(val, fallback):
                try:
                    return int(val)
                except Exception:
                    return fallback

            start = _safe_int(s, 0)
            end = _safe_int(e, start)
            if end < start:
                start, end = end, start

            empty = {
                "verb": "",
                "instrument_object_id": None,
                "target_object_id": None,
                "interaction_start": None,
                "functional_contact_onset": None,
                "interaction_end": None,
                "anomaly_label": self.extra_label_config.get("default_label", "Normal"),
            }
            hoi_data = {actor["id"]: dict(empty) for actor in self.actors_config}
            # Correct matching
            hand_key = None
            for actor in self.actors_config:
                if actor["id"].lower() == side_key:
                    hand_key = actor["id"]
                    break

            if hand_key:
                hoi_data[hand_key] = self._ensure_hand_annotation_state(
                    {
                    "verb": verb,
                    "instrument_object_id": tool_id,
                    "target_object_id": target_id,
                    "interaction_start": s,
                    "functional_contact_onset": o,
                    "interaction_end": e,
                    "anomaly_label": anomaly,
                    "_field_state": annotation_state.get("field_state", {}),
                    "_field_suggestions": annotation_state.get("field_suggestions", {}),
                    "_sparse_evidence_state": annotation_state.get(
                        "sparse_evidence_state", {}
                    ),
                    }
                )
            return {
                "event_id": self.event_id_counter,
                "frames": [start, end],
                "hoi_data": hoi_data,
            }

        for actor in self.actors_config:
            side_key = actor["id"].lower()
            for event in events.get(side_key, []) or []:
                self.events.append(_event_to_entry(side_key, event))
                self.event_id_counter += 1

        for event in self.events:
            hoi_data = event.get("hoi_data", {}) or {}
            for actor in self.actors_config:
                hand_key = actor["id"]
                if hand_key in hoi_data and isinstance(hoi_data[hand_key], dict):
                    self._hydrate_hand_annotation_state(
                        hoi_data[hand_key], default_source="loaded_annotation"
                    )
                    self._compute_sparse_evidence_state(hoi_data[hand_key])

        self._ensure_verbs_cover_events()
        self._refresh_events()
        self._after_events_loaded()

        video_id = data.get("video_id", "")
        QMessageBox.information(
            self,
            "Loaded",
            f"Loaded {len(self.events)} events.\nVideo ID: {video_id or 'unknown'}",
        )

    def _parse_yolo(self, path: str, w: int, h: int) -> List[dict]:
        """
        Supports:
        - Single txt with lines: frame cls xc yc w h [conf]
        - Or directory of per-frame txt (frame_XXXX.txt) with lines: cls xc yc w h [conf]
          coords assumed normalized if <=1, else treated as absolute center/size in px.
        """
        data: List[dict] = []
        if os.path.isdir(path):
            files = sorted([f for f in os.listdir(path) if f.lower().endswith(".txt")])
            for fname in files:
                frame_id = self._frame_from_filename(fname)
                full = os.path.join(path, fname)
                self._parse_yolo_file(full, w, h, data, frame_override=frame_id)
        else:
            self._parse_yolo_file(path, w, h, data, frame_override=None)
        return data

    def _frame_from_filename(self, fname: str) -> int:
        """Extract frame number from filename (e.g., frame_000123.txt -> 123)"""
        name = os.path.splitext(fname)[0]
        # expect frame_000123
        digits = "".join(ch for ch in name if ch.isdigit())
        try:
            return int(digits)
        except Exception:
            return 0

    def _parse_yolo_file(
        self, path: str, w: int, h: int, data: List[dict], frame_override: int = None
    ):
        """Parse individual YOLO text file."""
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                if len(parts) < 5:
                    continue
                idx = 0
                frame = None
                has_frame_token = len(parts) >= 6 and self._looks_number(parts[0])
                if has_frame_token:
                    try:
                        frame = int(float(parts[0]))
                        idx = 1
                    except Exception:
                        frame = None
                        idx = 0
                if frame is None:
                    frame = frame_override if frame_override is not None else 0
                cls = parts[idx]
                coords = parts[idx + 1 : idx + 5]
                if len(coords) < 4:
                    continue
                xc, yc, bw, bh = map(float, coords[:4])
                normed = max(xc, yc, bw, bh) <= 1.0
                if normed:
                    xc *= w
                    yc *= h
                    bw *= w
                    bh *= h
                x1 = max(0.0, xc - bw * 0.5)
                y1 = max(0.0, yc - bh * 0.5)
                x2 = min(float(w), xc + bw * 0.5)
                y2 = min(float(h), yc + bh * 0.5)
                box = {
                    "id": self.box_id_counter,
                    "orig_frame": frame,
                    "label": self._cls_name(cls),
                    "class_id": self._cls_id(cls),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
                self.box_id_counter += 1
                data.append(box)

    @staticmethod
    def _looks_number(txt: str) -> bool:
        try:
            float(txt)
            return True
        except Exception:
            return False

    def _cls_name(self, cls_token):
        try:
            cid = int(float(cls_token))
            if cid in self.class_map:
                return self.class_map[cid]
        except Exception:
            pass
        return str(cls_token)

    def _cls_id(self, cls_token):
        try:
            return int(float(cls_token))
        except Exception:
            return None

    def _build_category_to_default_id(self) -> dict:
        category_to_default_id = {}
        for obj_name, uid in self.global_object_map.items():
            category = self._norm_category(self.id_to_category.get(obj_name, obj_name))
            if not category:
                continue
            if (
                category not in category_to_default_id
                or uid < category_to_default_id[category]
            ):
                category_to_default_id[category] = uid
        return category_to_default_id

    def _norm_category(self, text: str) -> str:
        if not text:
            return ""
        cleaned = str(text).strip().lower()
        cleaned = cleaned.replace("-", "_").replace(" ", "_")
        cleaned = re.sub(r"_+", "_", cleaned)
        return cleaned.strip("_")

    def _parse_yaml_names(self, path: str) -> Dict[int, str]:
        """
        Minimal parser for data.yaml names mapping. Supports both:
        names: {0: 'label', 1: 'label2'} and flat mappings.
        """
        out: Dict = {}
        in_names = False
        pending_lines = []
        with open(path, "r", encoding="utf-8-sig") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                pending_lines.append(line)
                if line.startswith("names:"):
                    # Try to parse inline content
                    content = line[len("names:") :].strip()
                    if content.startswith("[") and content.endswith("]"):
                        items = content[1:-1].split(",")
                        for idx, itm in enumerate(items):
                            out[idx] = itm.strip().strip("'\"")
                        return out
                    if content.startswith("{") and content.endswith("}"):
                        pairs = content[1:-1].split(",")
                        for p in pairs:
                            if ":" in p:
                                k, v = p.split(":", 1)
                                try:
                                    out[int(k.strip())] = v.strip().strip("'\"")
                                except Exception:
                                    pass
                        return out
                    in_names = True
                    continue
                if in_names:
                    if ":" in line:
                        key_txt, val_txt = line.split(":", 1)
                        key_txt = key_txt.strip()
                        val_txt = val_txt.strip().strip("'\"")
                        try:
                            idx = int(key_txt)
                            out[idx] = val_txt
                        except Exception:
                            continue
                    else:
                        in_names = False
        if not out:
            for line in pending_lines:
                if ":" not in line:
                    continue
                key_txt, val_txt = line.split(":", 1)
                key_txt = key_txt.strip()
                val_txt = val_txt.strip().strip("'\"")
                if not key_txt.isdigit():
                    continue
                try:
                    idx = int(key_txt)
                    out[idx] = val_txt
                except Exception:
                    continue
        return out

    def _cid_from_label(self, label: str):
        """Return class id for a label name; auto-append if missing."""
        if label is None:
            return None
        for cid, name in self.class_map.items():
            if name == label:
                return cid
        if self.class_map:
            numeric_ids = []
            for k in self.class_map.keys():
                try:
                    numeric_ids.append(int(k))
                except Exception:
                    continue
            new_id = (max(numeric_ids) + 1) if numeric_ids else 0
            self.class_map[new_id] = label
            return new_id
        return None

    def _verb_id_by_name(self, name: str):
        for v in self.verbs:
            if v.name == name:
                return v.id
        return None

    def _replace_verb_in_hoi_data(self, hoi_data: dict, old: str, new: str) -> None:
        """Update verb string in a hoi_data dict for both hands."""
        if not hoi_data or not old or old == new:
            return
        for actor in self.actors_config:
            hand_key = actor["id"]
            h = hoi_data.get(hand_key, {})
            if h.get("verb") == old:
                h["verb"] = new

    def _collect_event_verbs(self) -> List[str]:
        """Return unique verb strings referenced by any hand in current events."""
        seen = []
        for ev in self.events:
            hoi_data = ev.get("hoi_data", {}) or {}
            for actor in self.actors_config:
                hand_key = actor["id"]
                verb = (hoi_data.get(hand_key, {}) or {}).get("verb") or ""
                if verb and verb not in seen:
                    seen.append(verb)
        return seen

    def _ensure_verbs_cover_events(self) -> None:
        """
        Ensure the verb list includes every verb referenced in events.
        Used when loading legacy files or events with verbs but no library.
        """
        name_to_id = {v.name: v.id for v in self.verbs}
        cur_max = max(name_to_id.values(), default=-1)
        added = False
        for verb in self._collect_event_verbs():
            if verb in name_to_id:
                continue
            cur_max += 1
            name_to_id[verb] = cur_max
            self.verbs.append(
                LabelDef(
                    name=verb, id=cur_max, color_name=self._color_for_index(cur_max)
                )
            )
            added = True
        if added:
            self._renumber_verbs()
            self._update_verb_combo()
            self.label_panel.refresh()

    def _build_payload_v2(self) -> dict:
        """[Fix] Build payload with STRICT filtering for empty hands."""

        self._refresh_sparse_evidence_snapshots()
        self.events.sort(key=lambda x: x.get("frames", [0, 0])[0])

        base = os.path.splitext(os.path.basename(self.video_path or "annotation"))[0]

        name_to_id = {v.name: v.id for v in self.verbs}
        cur_max_vid = max(name_to_id.values(), default=-1)
        for verb_name in self._collect_event_verbs():
            if verb_name in name_to_id:
                continue
            cur_max_vid += 1
            name_to_id[verb_name] = cur_max_vid
        verb_library = {
            str(v_id): name
            for name, v_id in sorted(name_to_id.items(), key=lambda x: x[1])
        }

        object_library = {}
        for name, uid in sorted(self.global_object_map.items(), key=lambda x: x[1]):
            entry = {"label": name, "category": name}
            cid = self._class_id_for_object(uid, name)
            if cid is not None:
                entry["class_id"] = cid
            object_library[str(uid)] = entry

        if not object_library:
            for rb in self.raw_boxes:
                label = rb.get("label")
                if self._is_hand_label(label):
                    continue
                uid = rb.get("id")
                if uid is None or str(uid) in object_library:
                    continue
                entry = {"label": str(label), "category": str(label)}
                cid = self._class_id_for_object(uid, label)
                if cid is not None:
                    entry["class_id"] = cid
                object_library[str(uid)] = entry

        def label_for_id(uid):
            for name, id_val in self.global_object_map.items():
                if id_val == uid:
                    return name
            return None

        # --------------------------------------

        # Build per-actor events
        events = {actor["id"].lower(): [] for actor in self.actors_config}
        counts = {actor["id"]: 0 for actor in self.actors_config}

        for i, event in enumerate(self.events):
            global_start, global_end = event.get("frames", [0, 0])

            for actor in self.actors_config:
                hand_key = actor["id"]
                side_key = hand_key.lower()
                h_data = event.get("hoi_data", {}).get(hand_key, {})
                self._ensure_hand_annotation_state(h_data)

                verb = h_data.get("verb", "")
                target = h_data.get("target_object_id")
                instr = h_data.get("instrument_object_id")
                s = h_data.get("interaction_start")
                o = h_data.get("functional_contact_onset")
                e = h_data.get("interaction_end")
                anom = h_data.get("anomaly_label", self.extra_label_config.get("default_label", "Normal"))

                has_verb = bool(verb and verb.strip())
                has_objects = (target is not None) or (instr is not None)
                has_timestamps = (s is not None) or (o is not None) or (e is not None)
                is_abnormal = anom != self.extra_label_config.get("default_label", "Normal")

                has_info = has_verb or has_objects or has_timestamps or is_abnormal

                if not has_info:
                    continue

                counts[hand_key] += 1
                prefix = self._get_actor_short_label(hand_key)
                event_id = f"{prefix}_{counts[hand_key]:03d}"

                final_start = s if s is not None else global_start
                final_onset = o if o is not None else global_start
                final_end = e if e is not None else global_end

                labels = self._parse_anomaly_labels(anom)
                allow_missing_verb = self._anomaly_rule_allows(
                    labels, "allow_missing_verb"
                )

                event_entry = {
                    "event_id": event_id,
                    "start_frame": final_start,
                    "contact_onset_frame": final_onset,
                    "end_frame": final_end,
                    "anomaly_label": anom,
                    "annotation_state": {
                        "field_state": copy.deepcopy(
                            h_data.get("_field_state", {}) or {}
                        ),
                        "field_suggestions": copy.deepcopy(
                            h_data.get("_field_suggestions", {}) or {}
                        ),
                        "sparse_evidence_state": copy.deepcopy(
                            h_data.get("_sparse_evidence_state", {}) or {}
                        ),
                    },
                }

                if has_verb:
                    event_entry["verb"] = verb
                elif not allow_missing_verb:
                    event_entry["verb"] = ""

                interaction = {}
                tool_label = label_for_id(instr)
                target_label = label_for_id(target)
                if tool_label:
                    interaction["tool"] = tool_label
                if target_label:
                    interaction["target"] = target_label
                event_entry["interaction"] = interaction

                links = {
                    "subject_track_id": f"T_{hand_key.upper()}",
                    "tool_track_id": f"T_OBJ_{instr}" if instr is not None else None,
                    "target_track_id": (
                        f"T_OBJ_{target}" if target is not None else None
                    ),
                }
                event_entry["links"] = links

                events[side_key].append(event_entry)

        tracks = {}

        def collect_hand_boxes(label):
            out = []
            for rb in self.raw_boxes:
                if rb.get("label") == label:
                    out.append(
                        {
                            "frame": rb["orig_frame"],
                            "bbox": [rb["x1"], rb["y1"], rb["x2"], rb["y2"]],
                        }
                    )
            out.sort(key=lambda x: x["frame"])
            return out

        for actor in self.actors_config:
            hand_key = actor["id"]
            side_key = hand_key.lower()
            track_id = f"T_{hand_key.upper()}"
            tracks[track_id] = {
                "category": side_key,
                "object_id": None,
                "boxes": collect_hand_boxes(hand_key),
            }

        for uid_str, info in object_library.items():
            uid = int(uid_str)
            label = info["label"]
            cat = info["category"]
            cid = info.get("class_id")
            b_list = []
            for rb in self.raw_boxes:
                if rb.get("id") == uid:
                    ent = {
                        "frame": rb["orig_frame"],
                        "bbox": [rb["x1"], rb["y1"], rb["x2"], rb["y2"]],
                    }
                    if cid is not None:
                        ent["class_id"] = cid
                    b_list.append(ent)
            b_list.sort(key=lambda x: x["frame"])
            tracks[f"T_OBJ_{uid}"] = {
                "category": cat,
                "object_id": uid,
                "boxes": b_list,
            }
            if cid is not None:
                tracks[f"T_OBJ_{uid}"]["class_id"] = cid

        return {
            "version": "HOI-1.0-ActionSeg",
            "video_id": base,
            "video_path": self.video_path or "",
            "fps": self.player.frame_rate,
            "frame_size": [self.player._frame_w, self.player._frame_h],
            "frame_count": self.player.frame_count,
            "bbox_mode": "xyxy",
            "bbox_normalized": False,
            "object_library": object_library,
            "verb_library": verb_library,
            "extra_label_config": self.extra_label_config,
            "actors_config": self.actors_config,
            "tracks": tracks,
            "hoi_events": events,
        }

    # ---------- UI refresh ----------
    def _sync_slider(self):
        # hooks are driven by VideoPlayer's callbacks; no extra slider here
        pass

    def _set_frame_controls(self, frame: int):
        super()._set_frame_controls(frame)
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.set_current_frame(frame)

    def _on_frame_advanced(self, frame: int):
        self._refresh_boxes_for_frame(frame)
        self._set_frame_controls(frame)

    def _on_player_playback_state_changed(self, _playing: bool):
        self._update_play_pause_button()

    def _update_play_pause_button(self):
        playing = bool(getattr(self.player, "is_playing", False))
        self.btn_play.setIcon(
            self.style().standardIcon(
                QStyle.SP_MediaPause if playing else QStyle.SP_MediaPlay
            )
        )
        self.btn_play.setToolTip("Pause" if playing else "Play")

    def _refresh_boxes_for_frame(self, frame: int, skip_events: bool = False):
        """
        Refresh logic: prepare boxes, update list, update player overlay, refresh relations.
        """
        self._update_overlay(frame)
        highlights = getattr(self, "_validation_highlights", {}) or {}
        highlight_ids = highlights.get("by_id", {})
        highlight_labels = highlights.get("by_label", {})
        boxes = self.bboxes.get(frame, [])
        self.current_hands = {actor["id"]: None for actor in self.actors_config}

        self.list_objects.blockSignals(True)
        self.list_objects.clear()

        display_boxes = []

        for b in boxes:
            lbl = str(b.get("label"))
            norm_hand = self._normalize_hand_label(lbl)
            if norm_hand:
                self.current_hands[norm_hand] = b
                hand_short = self._get_actor_short_label(norm_hand)
                item_txt = f"[Hand] {hand_short}"
                draw_label = hand_short
                color = "#ff00ff"
            else:
                uid = b.get("id")
                name = self._object_name_for_id(uid, fallback=f"ID_{uid}")
                if name.startswith("ID_") and b.get("label"):
                    name = str(b.get("label"))
                cid = b.get("class_id")
                if cid is None:
                    cid = self._class_id_for_label(b.get("label") or name)
                cid_txt = str(cid) if cid is not None else "_"
                item_txt = f"[obj:{uid} | cls:{cid_txt}] {name}"
                draw_label = item_txt
                color = self._class_color(cid)

            item = QListWidgetItem(item_txt)
            item.setData(Qt.UserRole, b)
            self.list_objects.addItem(item)
            thick = False
            if self.validation_enabled:
                if b.get("id") in highlight_ids:
                    color = highlight_ids[b.get("id")]
                    thick = True
                elif norm_hand and norm_hand in highlight_labels:
                    color = highlight_labels[norm_hand]
                    thick = True

            display_boxes.append(
                {
                    "x1": b["x1"],
                    "y1": b["y1"],
                    "x2": b["x2"],
                    "y2": b["y2"],
                    "label": draw_label,
                    "color": color,
                    "thick": thick,
                }
            )

        self.list_objects.blockSignals(False)

        if hasattr(self.player, "set_overlay_boxes"):
            self.player.set_overlay_boxes(display_boxes)
        elif hasattr(self.player, "set_boxes"):
            self.player.set_boxes(display_boxes)
        else:
            print("Warning: VideoPlayer missing set_overlay_boxes method")

        if hasattr(self.player, "set_edit_context"):
            if self.chk_edit_boxes.isChecked():
                edit_boxes = []
                for b in boxes:
                    edit_boxes.append(
                        {
                            "id": b.get("id"),
                            "frame": frame,
                            "orig_frame": b.get("orig_frame"),
                            "x1": b.get("x1"),
                            "y1": b.get("y1"),
                            "x2": b.get("x2"),
                            "y2": b.get("y2"),
                            "label": b.get("label"),
                            "class_id": b.get("class_id"),
                        }
                    )
                self.player.set_edit_context(
                    edit_boxes,
                    on_change=self._on_box_edited,
                    label_resolver=self._resolve_label_and_id,
                    label_suggestions=self._label_suggestions(),
                    auto_label_fetcher=self._get_auto_draw_label,
                )
            else:
                self.player.set_edit_context(
                    [],
                    on_change=None,
                    label_resolver=None,
                    auto_label_fetcher=None,
                )

        self._update_inspector_tab_labels()

        if not skip_events:
            self._refresh_events(refresh_boxes=False)

    def _get_auto_draw_label(self):
        if not self.selected_hand_label:
            return None
        hand_data = self.event_draft.get(self.selected_hand_label) or {}
        target_obj_id = None
        if getattr(self, "rad_draw_inst", None) and self.rad_draw_inst.isChecked():
            target_obj_id = hand_data.get("instrument_object_id")
        elif getattr(self, "rad_draw_target", None) and self.rad_draw_target.isChecked():
            target_obj_id = hand_data.get("target_object_id")
        if target_obj_id is None:
            return None
        for name, uid in self.global_object_map.items():
            if uid == target_obj_id:
                return name, target_obj_id
        return None

    def _refresh_events(self, refresh_boxes: bool = True):
        """[Step 3 Fix] Refresh list: show detailed info including Instrument for error checking."""
        frame = self.player.current_frame
        self._refresh_sparse_evidence_snapshots()
        for ev in self.events:
            self._sync_event_frames(ev)
        self._update_overlay(frame)
        if refresh_boxes and self.validation_enabled:
            self._refresh_boxes_for_frame(frame, skip_events=True)
        self._update_incomplete_indicator()
        self._update_next_best_query_panel()
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.refresh()

    def _update_overlay(self, frame: int):
        """
        [Step 2 Fix] Draw Hand -> Instrument -> Target relation lines on video.
        Adapted to self.events structure.
        """
        hand_color = "#3b82f6"
        instrument_color = "#f59e0b"
        target_color = "#22c55e"
        display_rels = []

        # 1. Valid HOIs from committed events
        for ev in self.events:
            for actor in self.actors_config:
                hand_key = actor["id"]
                h_data = ev["hoi_data"][hand_key]
                s = h_data.get("interaction_start")
                e = h_data.get("interaction_end")

                if s is not None and e is not None and s <= frame <= e:
                    draw_item = dict(h_data)
                    draw_item["hand_id"] = hand_key
                    display_rels.append(draw_item)

        # 2. Preview from current Draft
        if hasattr(self, "event_draft"):
            for actor in self.actors_config:
                hand_key = actor["id"]
                h_data = self.event_draft[hand_key]
                if h_data.get("target_object_id"):
                    draw_item = dict(h_data)
                    draw_item["hand_id"] = hand_key
                    display_rels.append(draw_item)

        overlay_data = []
        highlight_ids = {}
        highlight_labels = {}
        current_boxes = {b["id"]: b for b in self.bboxes.get(frame, [])}

        for r in display_rels:
            hand_label = r.get("hand_id")
            hand_box = None
            for b in current_boxes.values():
                if self._normalize_hand_label(b.get("label")) == hand_label:
                    hand_box = b
                    break

            if not hand_box:
                continue

            inst_id = r.get("instrument_object_id")
            target_id = r.get("target_object_id")
            verb = r.get("verb", "unknown")

            # Case A: Only Target
            if inst_id is None and target_id is not None:
                if target_id in current_boxes:
                    overlay_data.append(
                        {
                            "box_a": hand_box,
                            "box_b": current_boxes[target_id],
                            "label": verb,
                            "color": target_color,
                        }
                    )
                    if self.validation_enabled:
                        highlight_labels[hand_label] = hand_color
                        highlight_ids[target_id] = target_color

            # Case B: With Instrument
            elif inst_id is not None:
                if inst_id in current_boxes:
                    # Line 1: Hand -> Instrument
                    inst_box = current_boxes[inst_id]
                    overlay_data.append(
                        {
                            "box_a": hand_box,
                            "box_b": inst_box,
                            "label": "",
                            "color": instrument_color,
                        }
                    )
                    if self.validation_enabled:
                        highlight_labels[hand_label] = hand_color
                        highlight_ids[inst_id] = instrument_color
                    # Line 2: Instrument -> Target
                    if target_id is not None and target_id in current_boxes:
                        overlay_data.append(
                            {
                                "box_a": inst_box,
                                "box_b": current_boxes[target_id],
                                "label": verb,
                                "color": target_color,
                            }
                        )
                        if self.validation_enabled:
                            highlight_ids[target_id] = target_color

        if self.validation_enabled:
            self._validation_highlights = {
                "by_id": highlight_ids,
                "by_label": highlight_labels,
            }
        else:
            self._validation_highlights = {}

        overlay_lines = []
        for rel in overlay_data:
            try:
                box_a = rel["box_a"]
                box_b = rel["box_b"]
                x1 = (float(box_a["x1"]) + float(box_a["x2"])) / 2.0
                y1 = (float(box_a["y1"]) + float(box_a["y2"])) / 2.0
                x2 = (float(box_b["x1"]) + float(box_b["x2"])) / 2.0
                y2 = (float(box_b["y1"]) + float(box_b["y2"])) / 2.0
            except Exception:
                continue
            overlay_lines.append(
                {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "label": rel.get("label", ""),
                    "color": rel.get("color", ""),
                }
            )
        self.player.set_overlay_relations(overlay_lines)

    def _stop(self):
        try:
            self.player.stop()
        except Exception:
            pass
        self._update_play_pause_button()
        self._set_frame_controls(self.player.current_frame)
        self._refresh_boxes_for_frame(self.player.current_frame)
        self._log("hoi_stop")

    def _toggle_play_pause(self):
        if getattr(self.player, "is_playing", False):
            self._pause()
        else:
            self._play()

    def _play(self):
        try:
            self.player.play()
        except Exception:
            pass
        self._update_play_pause_button()
        self._log("hoi_play")

    def _pause(self):
        try:
            self.player.pause()
        except Exception:
            pass
        self._update_play_pause_button()
        self._log("hoi_pause")

    def _seek_relative(self, delta: int):
        if not self.player.cap:
            return
        target = max(
            self.player.crop_start,
            min(self.player.crop_end, self.player.current_frame + delta),
        )
        self.player.seek(target)
        self._refresh_boxes_for_frame(target)
        self._set_frame_controls(target)
        self._log("hoi_seek_relative", delta=delta, target=target)

    def _seek_seconds(self, direction: int):
        """Seek by +/-1 second based on video FPS."""
        if not self.player.cap:
            return
        fps = max(1, int(round(self.player.frame_rate or 30)))
        self._seek_relative(direction * fps)

    def _jump_to_spin(self):
        frame = int(self.spin_jump.value())
        if self.player.cap:
            self.player.seek(frame)
            self._refresh_boxes_for_frame(frame)
            self.slider.blockSignals(True)
            self.slider.setValue(frame)
            self.slider.blockSignals(False)
            if getattr(self, "hoi_timeline", None):
                self.hoi_timeline.set_current_frame(frame)
            self._log("hoi_jump_to_frame", target=frame)

    def _on_slider_changed(self, val: int):
        if self.player.cap:
            self.player.seek(val)
            self._refresh_boxes_for_frame(val)
            if getattr(self, "hoi_timeline", None):
                self.hoi_timeline.set_current_frame(val)
            self._log("hoi_seek_slider", target=val)

    def _on_offset_changed(self):
        self.start_offset = int(self.spin_start_offset.value())
        end_val = int(self.spin_end_frame.value())
        self.end_frame = end_val if end_val >= self.start_offset else None
        self._rebuild_bboxes_from_raw()
        self._refresh_boxes_for_frame(self.player.current_frame)
        self._log("hoi_crop_set", start=self.start_offset, end=self.end_frame)

    def _on_edit_boxes_toggled(self, on: bool):
        for widget in (
            getattr(self, "rad_draw_none", None),
            getattr(self, "rad_draw_inst", None),
            getattr(self, "rad_draw_target", None),
        ):
            if widget is not None:
                widget.setEnabled(bool(on))
        if not on and getattr(self, "rad_draw_none", None) is not None:
            self.rad_draw_none.setChecked(True)
        self._update_draw_mode_visibility()
        self._refresh_boxes_for_frame(self.player.current_frame)
        self._log("hoi_edit_boxes_toggle", on=bool(on))

    def _set_validation_ui_state(self, on: bool):
        if getattr(self, "lbl_validation", None):
            if on:
                self.lbl_validation.setStyleSheet("color: #12b76a; font-weight: 600;")
            else:
                self.lbl_validation.setStyleSheet("")
        self._set_status_chip(
            getattr(self, "lbl_validation_chip", None),
            "Validation On" if on else "Validation Off",
            "ok" if on else "neutral",
        )
        if getattr(self, "btn_validation", None):
            if on:
                self.btn_validation.setToolTip("Return to annotation mode")
            else:
                self.btn_validation.setToolTip("Toggle validation on/off")
        self._update_incomplete_indicator()

    def _on_validation_toggled(self, on: bool):
        if on:
            name, ok = QInputDialog.getText(self, "Validation", "Editor name:")
            if not ok or not name.strip():
                self.btn_validation.blockSignals(True)
                self.btn_validation.setChecked(False)
                self.btn_validation.blockSignals(False)
                self._set_validation_ui_state(False)
                self._log("hoi_validation_cancel")
                return
            self.validator_name = name.strip()
            self.validation_enabled = True
            self.validation_session_active = True
            self.validation_started_at = datetime.now().isoformat(timespec="seconds")
            self.validation_modified = False
            self.validation_change_count = 0
            if getattr(self, "validation_op_logger", None):
                self.validation_op_logger.enabled = self.op_logger.enabled
                self.validation_op_logger.clear()
            self._set_validation_ui_state(True)
            self._refresh_boxes_for_frame(self.player.current_frame)
            self._log("hoi_validation_on", validator=self.validator_name)
        else:
            self.validation_enabled = False
            self._set_validation_ui_state(False)
            self._refresh_boxes_for_frame(self.player.current_frame)
            self._log("hoi_validation_off")

    def _on_task_combo_changed(self, text: str):
        lower = (text or "").lower()
        if ("handoi" in lower) or ("hoi" in lower):
            return
        if callable(self._on_switch_task):
            self._on_switch_task(text)

    def _experiment_mode_key(self) -> str:
        combo = getattr(self, "combo_experiment_mode", None)
        if combo is not None:
            return str(combo.currentData() or getattr(self, "_experiment_mode", "full_assist"))
        return str(getattr(self, "_experiment_mode", "full_assist") or "full_assist")

    def _detection_assist_enabled(self) -> bool:
        return self._experiment_mode_key() in ("assist", "full_assist")

    def _semantic_assist_enabled(self) -> bool:
        return self._experiment_mode_key() == "full_assist"

    def _guard_experiment_mode(self, feature: str) -> bool:
        if feature == "detection" and self._detection_assist_enabled():
            return True
        if feature == "semantic" and self._semantic_assist_enabled():
            return True
        if feature == "semantic":
            QMessageBox.information(
                self,
                "Experiment Mode",
                "Switch to Full Assist to use VideoMAE-based action-label assistance.",
            )
        else:
            QMessageBox.information(
                self,
                "Experiment Mode",
                "Switch to Assist or Full Assist to use imported or detected assistance.",
            )
        return False

    def _apply_experiment_mode_ui(self) -> None:
        mode = self._experiment_mode_key()
        self._experiment_mode = mode
        detection_enabled = self._detection_assist_enabled()
        semantic_enabled = self._semantic_assist_enabled()

        for obj in (
            getattr(self, "act_import_yolo_boxes", None),
            getattr(self, "act_load_yolo_model", None),
            getattr(self, "act_detect_current_frame", None),
            getattr(self, "act_detect_selected_action", None),
            getattr(self, "act_detect_all_actions", None),
            getattr(self, "act_load_hands_xml", None),
            getattr(self, "act_incremental_train_yolo", None),
            getattr(self, "act_swap_hands", None),
            getattr(self, "btn_detect", None),
            getattr(self, "btn_detect_action", None),
            getattr(self, "btn_object_tools", None),
        ):
            if obj is not None:
                obj.setEnabled(detection_enabled)

        for obj in (
            getattr(self, "act_load_videomae_model", None),
            getattr(self, "act_load_videomae_verb_list", None),
            getattr(self, "act_review_selected_action_label", None),
            getattr(self, "act_auto_apply_action_labels", None),
            getattr(self, "act_batch_apply_action_labels", None),
            getattr(self, "btn_suggest_action_label", None),
        ):
            if obj is not None:
                obj.setEnabled(semantic_enabled)

        if getattr(self, "btn_suggest_action_label", None) is not None:
            self.btn_suggest_action_label.setVisible(semantic_enabled)
        if getattr(self, "act_batch_apply_action_labels", None) is not None:
            self.act_batch_apply_action_labels.setVisible(semantic_enabled)

        if not semantic_enabled:
            try:
                self._videomae_auto_timer.stop()
            except Exception:
                pass
            self._update_action_top5_display(None)
        else:
            self._update_action_top5_display(self.selected_event_id)
            if self.selected_event_id is not None:
                self._queue_action_label_refresh(self.selected_event_id, delay_ms=120)

    def _on_experiment_mode_changed(self, _index: int) -> None:
        self._apply_experiment_mode_ui()
        self._log("hoi_experiment_mode_changed", mode=self._experiment_mode_key())

    def _hoi_state_signature(self) -> str:
        payload = {
            "events": self.events,
            "raw_boxes": self.raw_boxes,
            "event_draft": getattr(self, "event_draft", None),
            "object_map": self.global_object_map,
            "class_map": self.class_map,
            "start_offset": self.start_offset,
            "end_frame": self.end_frame,
        }
        try:
            blob = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
        except Exception:
            blob = str(
                (
                    len(self.events),
                    len(self.raw_boxes),
                    self.start_offset,
                    self.end_frame,
                )
            )
        return hashlib.md5(blob.encode("utf-8")).hexdigest()

    def _mark_hoi_saved(self) -> None:
        self._hoi_saved_signature = self._hoi_state_signature()

    def _hoi_has_unsaved_changes(self) -> bool:
        sig = self._hoi_state_signature()
        if self._hoi_saved_signature is None:
            has_data = bool(
                self.events or self.raw_boxes or getattr(self, "event_draft", None)
            )
            return has_data
        return sig != self._hoi_saved_signature

    def _confirm_task_switch(self, _target: str = "") -> bool:
        if not self._hoi_has_unsaved_changes():
            return True
        msg = "HandOI has unsaved changes.\nSwitch tasks anyway?"
        reply = QMessageBox.question(
            self,
            "Unsaved HandOI changes",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    def _on_object_selection(self):
        if not self.selected_hand_label:
            self._clear_relation_highlight()
            self._update_overlay(self.player.current_frame)
            return

        items = self.list_objects.selectedItems()
        if not items:
            return
        obj_box = items[0].data(Qt.UserRole)
        if not obj_box:
            return
        if self._is_hand_label(obj_box.get("label")):
            return
        obj_id = obj_box["id"]

        hand_data = self.event_draft.get(self.selected_hand_label)
        if not hand_data:
            return
        prev_inst = hand_data["instrument_object_id"]
        prev_tar = hand_data["target_object_id"]

        if hand_data["instrument_object_id"] is None:
            hand_data["instrument_object_id"] = obj_id
            self._set_hand_field_state(
                hand_data,
                "instrument_object_id",
                source="object_pick",
                status="confirmed",
            )
        elif hand_data["target_object_id"] is None:
            if obj_id == hand_data["instrument_object_id"]:
                hand_data["instrument_object_id"] = None
                self._clear_hand_field(
                    hand_data, "instrument_object_id", source="object_pick_reassign"
                )
                hand_data["target_object_id"] = obj_id
                self._set_hand_field_state(
                    hand_data,
                    "target_object_id",
                    source="object_pick",
                    status="confirmed",
                )
            else:
                hand_data["target_object_id"] = obj_id
                self._set_hand_field_state(
                    hand_data,
                    "target_object_id",
                    source="object_pick",
                    status="confirmed",
                )

        self._load_hand_draft_to_ui(self.selected_hand_label)
        self._update_overlay(self.player.current_frame)
        self._apply_draft_to_selected_event()
        self._refresh_events()
        self._update_hoi_titles()
        self.hoi_timeline.refresh()
        if (
            prev_inst != hand_data["instrument_object_id"]
            or prev_tar != hand_data["target_object_id"]
        ):
            self._log(
                "hoi_select_object",
                hand=self.selected_hand_label,
                instrument=hand_data["instrument_object_id"],
                target=hand_data["target_object_id"],
            )

    def _select_hand(self, hand_label: str, on: bool):
        """[Step 2] Switch hand context: save old data -> switch -> load new data."""

        # 1. Mutex Logic & UI State Management
        if on:
            # Save previous hand state
            if self.selected_hand_label and self.selected_hand_label != hand_label:
                self._save_ui_to_hand_draft(self.selected_hand_label)
                self._apply_draft_to_selected_event()

            # Set new state
            self.selected_hand_label = hand_label

            # Enforce Mutex UI: uncheck other actors
            for aid, chk in self.actor_controls.items():
                if aid != hand_label:
                    chk.blockSignals(True)
                    chk.setChecked(False)
                    chk.blockSignals(False)

            # Load new hand data to UI
            self._load_hand_draft_to_ui(hand_label)
            self._log("hoi_select_hand", hand=hand_label, on=True)
        else:
            # Uncheck current hand
            if self.selected_hand_label == hand_label:
                self._save_ui_to_hand_draft(hand_label)  # Save final state
                self._apply_draft_to_selected_event()
                self.selected_hand_label = None
                self._update_status_label()
                self._log("hoi_select_hand", hand=hand_label, on=False)

        self.list_objects.setEnabled(self.selected_hand_label is not None)
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.set_selected(
                self.selected_event_id, self.selected_hand_label
            )
            self._update_hoi_titles()
        self._update_overlay(self.player.current_frame)

    def _clear_relation_highlight(self):
        self.selected_event_id = None
        self.selected_hand_label = None
        self._reset_event_draft()
        if getattr(self, "hoi_timeline", None):
            self.hoi_timeline.set_selected(None, None)
            self._update_hoi_titles()
        self._update_overlay(self.player.current_frame)

    def _on_box_edited(self, box_id: int, new_box: dict):
        """Apply edits to the raw_boxes by id and refresh views."""
        action = new_box.get("_action")
        target_frame = new_box.get("frame")
        target_orig = new_box.get("orig_frame")
        if target_orig is None and target_frame is not None:
            try:
                target_orig = int(target_frame) - int(self.start_offset)
            except Exception:
                target_orig = None

        def _matches(rb):
            if box_id is not None and rb.get("id") != box_id:
                return False
            if target_orig is not None and rb.get("orig_frame") != target_orig:
                return False
            return True

        if action == "delete":
            if any(_matches(rb) for rb in self.raw_boxes):
                self._push_undo()
                self.raw_boxes = [rb for rb in self.raw_boxes if not _matches(rb)]
                self._rebuild_bboxes_from_raw()
                self._refresh_boxes_for_frame(self.player.current_frame)
                self._log(
                    "hoi_box_delete", box_id=box_id, frame=self.player.current_frame
                )
            return

        label_txt = new_box.get("label")
        new_class_id = new_box.get("class_id", None)
        if label_txt is not None:
            lbl_res, cid_res, _ = self._resolve_label_and_id(label_txt)
            if lbl_res:
                label_txt = lbl_res
            if cid_res is not None:
                new_class_id = cid_res

        if action == "add":
            if not label_txt:
                return

            obj_id = None
            if self.global_object_map and self.id_to_category:
                norm_label = self._norm_category(label_txt)
                if norm_label:
                    category_to_default_id = self._build_category_to_default_id()
                    obj_id = category_to_default_id.get(norm_label)

            if target_orig is None:
                try:
                    target_orig = int(self.player.current_frame) - int(
                        self.start_offset
                    )
                except Exception:
                    target_orig = 0

            is_new_object = False
            if obj_id is None:
                obj_id = self.object_id_counter
                is_new_object = True

            self._push_undo()
            if new_class_id is not None and label_txt:
                if (
                    new_class_id not in self.class_map
                    and str(new_class_id) not in self.class_map
                ):
                    self.class_map[new_class_id] = label_txt
            if is_new_object:
                self.object_id_counter += 1
                self._register_object_entry(obj_id, label_txt)

            new_rb = {
                "id": obj_id,
                "orig_frame": target_orig,
                "label": label_txt,
                "source": "manual_box_add",
                "x1": new_box.get("x1"),
                "y1": new_box.get("y1"),
                "x2": new_box.get("x2"),
                "y2": new_box.get("y2"),
            }
            if new_class_id is not None:
                new_rb["class_id"] = new_class_id

            self.raw_boxes.append(new_rb)
            self._rebuild_bboxes_from_raw()
            self._refresh_boxes_for_frame(self.player.current_frame)
            self._log(
                "hoi_box_add",
                box_id=obj_id,
                class_id=new_class_id,
                label=label_txt,
                frame=self.player.current_frame,
            )
            return

        changed = False
        if any(_matches(rb) for rb in self.raw_boxes):
            self._push_undo()
        if new_class_id is not None and label_txt:
            if (
                new_class_id not in self.class_map
                and str(new_class_id) not in self.class_map
            ):
                self.class_map[new_class_id] = label_txt
        for rb in self.raw_boxes:
            if _matches(rb):
                for key in ("x1", "y1", "x2", "y2"):
                    if key in new_box and new_box[key] is not None:
                        rb[key] = new_box[key]
                if label_txt is not None:
                    rb["label"] = label_txt
                if new_class_id is not None:
                    rb["class_id"] = new_class_id
                rb["source"] = "manual_box_edit"
                changed = True
        if changed:
            self._rebuild_bboxes_from_raw()
            self._refresh_boxes_for_frame(self.player.current_frame)
            self._log(
                "hoi_box_edit",
                box_id=box_id,
                class_id=new_class_id,
                label=label_txt,
                frame=self.player.current_frame,
            )

    def _resolve_label_and_id(self, text: str):
        """
        Resolve text to (label, class_id, known_flag).
        Matches numeric ids or class names.
        """
        if text is None:
            return "", None, False
        stripped = str(text).strip()
        try:
            cid = int(stripped)
            if cid in self.class_map:
                return self.class_map[cid], cid, True
            if str(cid) in self.class_map:
                return self.class_map[str(cid)], cid, True
            for rb in self.raw_boxes:
                if rb.get("class_id") == cid and rb.get("label"):
                    return str(rb.get("label")), cid, True
            return stripped, cid, False
        except Exception:
            pass
        lowered = stripped.lower()
        for k, v in self.class_map.items():
            try:
                if str(v).lower() == lowered:
                    try:
                        cid_val = int(k)
                    except Exception:
                        cid_val = k
                    return v, cid_val, True
            except Exception:
                continue
        for rb in self.raw_boxes:
            if str(rb.get("label", "")).lower() == lowered:
                return rb.get("label"), rb.get("class_id"), True
        return stripped, None, False

    def _class_id_for_label(self, label: str):
        if not label:
            return None
        target = str(label).lower().strip()
        for key, val in self.class_map.items():
            try:
                if str(val).lower().strip() == target:
                    try:
                        return int(key)
                    except Exception:
                        return key
            except Exception:
                continue
        return None

    def _class_id_for_object(self, uid, label=None):
        for rb in self.raw_boxes:
            if rb.get("id") == uid and rb.get("label") not in (
                self.actors_config[0]["id"],
                self.actors_config[1]["id"] if len(self.actors_config) > 1 else self.actors_config[0]["id"],
            ):
                cid = rb.get("class_id")
                if cid is not None:
                    return cid
        if label:
            return self._class_id_for_label(label)
        return None

    def _class_color(self, class_id):
        if class_id is None:
            return "#00ffff"
        try:
            idx = abs(int(class_id))
        except Exception:
            idx = abs(hash(str(class_id)))
        color_key = self._color_for_index(idx)
        return color_from_key(color_key).name()

    def _object_name_for_id(
        self, uid, default_for_none: str = "_", fallback: str = None
    ) -> str:
        if uid is None:
            return default_for_none
        for name, id_val in self.global_object_map.items():
            if id_val == uid:
                return name
        if fallback is not None:
            return fallback
        return str(uid)

    def _register_object_entry(self, uid, label: str):
        """Ensure a new object id is represented in the registry and combos."""
        if uid is None:
            return
        for name, id_val in self.global_object_map.items():
            if id_val == uid:
                return
        base = (label or f"obj_{uid}").strip()
        if not base:
            base = f"obj_{uid}"
        name = base
        if name in self.global_object_map:
            idx = 1
            while f"{base}_{idx}" in self.global_object_map:
                idx += 1
            name = f"{base}_{idx}"
        self.global_object_map[name] = uid
        self.id_to_category[name] = base
        display_text = f"[{uid}] {name}"
        if self.combo_instrument.findData(uid) == -1:
            self.combo_instrument.addItem(display_text, uid)
        if self.combo_target.findData(uid) == -1:
            self.combo_target.addItem(display_text, uid)
        if uid >= self.object_id_counter:
            self.object_id_counter = uid + 1

    def _rebuild_object_combos(self, inst_selected=None, tgt_selected=None):
        """Rebuild instrument/target combos from the current object library."""
        self.combo_instrument.blockSignals(True)
        self.combo_target.blockSignals(True)
        self.combo_instrument.clear()
        self.combo_target.clear()
        self.combo_instrument.addItem("None", None)
        self.combo_target.addItem("None", None)
        for name, uid in sorted(self.global_object_map.items(), key=lambda x: x[1]):
            display_text = f"[{uid}] {name}"
            self.combo_instrument.addItem(display_text, uid)
            self.combo_target.addItem(display_text, uid)
        if inst_selected is not None:
            idx = self.combo_instrument.findData(inst_selected)
            if idx >= 0:
                self.combo_instrument.setCurrentIndex(idx)
        if tgt_selected is not None:
            idx = self.combo_target.findData(tgt_selected)
            if idx >= 0:
                self.combo_target.setCurrentIndex(idx)
        self.combo_instrument.blockSignals(False)
        self.combo_target.blockSignals(False)

    def _label_suggestions(self):
        suggestions = []
        for cid, name in self.class_map.items():
            suggestions.append(str(cid))
            if name:
                suggestions.append(str(name))
        seen = set()
        ordered = []
        for item in suggestions:
            if item and item not in seen:
                seen.add(item)
                ordered.append(item)
        return ordered

    def closeEvent(self, e):
        ok_incomplete, _ = self._check_incomplete_hoi(context="close")
        if not ok_incomplete:
            e.ignore()
            return
        if callable(self._on_close):
            try:
                self._on_close()
            except Exception:
                pass
        return super().closeEvent(e)

    def _rebuild_bboxes_from_raw(self):
        self.bboxes = {}
        for b in self.raw_boxes:
            target_frame = b.get("orig_frame", 0) + self.start_offset
            if self.end_frame is not None and target_frame > self.end_frame:
                continue
            mapped = dict(b)
            mapped["frame"] = target_frame
            self.bboxes.setdefault(target_frame, []).append(mapped)

    def _load_hands_xml(self):
        """[Fix] Load hand bbox XML with Auto-Alignment for clipped videos."""
        if not self._guard_experiment_mode("detection"):
            return
        fp, _ = QFileDialog.getOpenFileName(
            self, "Load hand bbox XML", "", "XML Files (*.xml);;All Files (*)"
        )
        if not fp:
            return

        if not self.player.cap:
            QMessageBox.information(self, "Info", "Please load a video first.")
            return

        try:
            parsed = self._parse_cvat_xml(fp)

            if not parsed:
                QMessageBox.warning(self, "Parse Error", "No valid hand boxes found.")
                return

            min_xml_frame = min(p["orig_frame"] for p in parsed)
            max_xml_frame = max(p["orig_frame"] for p in parsed)
            video_frame_count = self.player.frame_count
            offset_needed = 0

            if min_xml_frame > 0:
                msg = (
                    f"XML data starts at frame {min_xml_frame} (End: {max_xml_frame}).\n"
                    f"Video length is {video_frame_count} frames.\n\n"
                    f"Do you want to AUTO-ALIGN the data?\n"
                    f"(Shift frames by -{min_xml_frame}, so Frame {min_xml_frame} becomes Frame 0)"
                )

                reply = QMessageBox.question(
                    self, "Auto Align?", msg, QMessageBox.Yes | QMessageBox.No
                )

                if reply == QMessageBox.Yes:
                    offset_needed = min_xml_frame
                    for p in parsed:
                        p["orig_frame"] -= offset_needed

                    max_xml_frame -= offset_needed
                    print(
                        f"DEBUG: Applied offset -{offset_needed}. New range: 0 to {max_xml_frame}"
                    )

            final_max_frame = max_xml_frame + self.start_offset

            if final_max_frame >= video_frame_count:
                QMessageBox.warning(
                    self,
                    "Warning",
                    f"Hand bbox frames exceed video length!\n\n"
                    f"Max Data Frame: {final_max_frame}\n"
                    f"Video Length: {video_frame_count}\n\n"
                    "If you chose 'No' to alignment, try loading again and verify offsets.",
                )

        except Exception as ex:
            import traceback

            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to parse XML:\n{ex}")
            return

        self._push_undo()

        self.raw_boxes = [
            b for b in self.raw_boxes if not self._is_hand_label(b.get("label"))
        ]

        self.raw_boxes.extend(parsed)
        self._rebuild_bboxes_from_raw()
        self._refresh_boxes_for_frame(self.player.current_frame)

        success_msg = (
            f"Loaded {len(parsed)} hand boxes.\nOriginal Start: {min_xml_frame}"
        )
        if offset_needed > 0:
            success_msg += f"\nAuto-Aligned: shifted by -{offset_needed} frames."

        QMessageBox.information(self, "Success", success_msg)
        self._log(
            "hoi_load_hands_xml",
            path=fp,
            count=len(parsed),
            offset_shifted=offset_needed,
        )
        self._mark_hoi_saved()

    def _parse_cvat_xml(self, path: str) -> List[dict]:
        """
        [Updated] Robust parser for CVAT XML.
        - Supports 'CVAT for Images' (your current format).
        - Supports 'CVAT for Video' (legacy format).
        - Auto-corrects labels: "left"->"Left_hand", "right"->"Right_hand".
        - Extracts frame numbers from filenames like "...-00135.jpg".
        """
        import xml.etree.ElementTree as ET
        import re

        try:
            tree = ET.parse(path)
            root = tree.getroot()
        except Exception as e:
            raise ValueError(f"Invalid XML file: {e}")

        out = []

        def normalize_label(l_str):
            if not l_str:
                return None
            l_lower = str(l_str).lower().strip()
            if l_lower in ["left", "left_hand", "l_hand", "left hand"]:
                return self.actors_config[0]["id"]
            if l_lower in ["right", "right_hand", "r_hand", "right hand"]:
                return self.actors_config[1]["id"] if len(self.actors_config) > 1 else self.actors_config[0]["id"]
            return None

        def extract_frame_from_name(name_str):
            matches = re.findall(r"\d+", name_str)
            if matches:
                return int(matches[-1])
            return 0

        images = root.findall("image")
        if images:
            print(
                f"Debug: Found {len(images)} image tags. Parsing 'CVAT for Images' format..."
            )
            for img in images:
                name = img.attrib.get("name", "")
                img_id = img.attrib.get("id")

                try:
                    frame = extract_frame_from_name(name)
                except Exception:
                    frame = int(img_id) if img_id else 0

                for box in img.findall("box"):
                    raw_label = box.attrib.get("label")
                    label = normalize_label(raw_label)

                    if label is None:
                        continue

                    try:
                        out.append(
                            {
                                "id": self.box_id_counter,
                                "orig_frame": frame,
                                "label": label,
                                "source": "hands_xml",
                                "x1": float(box.attrib.get("xtl")),
                                "y1": float(box.attrib.get("ytl")),
                                "x2": float(box.attrib.get("xbr")),
                                "y2": float(box.attrib.get("ybr")),
                            }
                        )
                        self.box_id_counter += 1
                    except (ValueError, TypeError):
                        continue

            if out:
                return out

        tracks = root.findall("track")
        if tracks:
            print(
                f"Debug: Found {len(tracks)} track tags. Parsing 'CVAT for Video' format..."
            )
            for track in tracks:
                raw_label = track.attrib.get("label")
                label = normalize_label(raw_label)
                if label is None:
                    continue

                for box in track.findall("box"):
                    if box.attrib.get("outside") == "1":
                        continue
                    try:
                        out.append(
                            {
                                "id": self.box_id_counter,
                                "orig_frame": int(box.attrib.get("frame")),
                                "label": label,
                                "source": "hands_xml",
                                "x1": float(box.attrib.get("xtl")),
                                "y1": float(box.attrib.get("ytl")),
                                "x2": float(box.attrib.get("xbr")),
                                "y2": float(box.attrib.get("ybr")),
                            }
                        )
                        self.box_id_counter += 1
                    except Exception:
                        continue

        return out

    def _validate_integrity(self, relation: dict) -> list:
        """
        Check BBox existence for Start/Onset/End frames in a Relation.
        Returns error list: [{'msg': '...', 'frame': 123}, ...]
        """
        errors = []

        check_points = {
            "Start": relation.get("interaction_start"),
            "Onset": relation.get("functional_contact_onset"),
            "End": relation.get("interaction_end"),
        }

        targets = [
            ("Instrument", relation.get("instrument_object_id")),
            ("Target", relation.get("target_object_id")),
        ]

        for time_label, frame in check_points.items():
            if frame is None:
                continue

            frame_boxes = self.bboxes.get(frame, [])
            existing_ids = set(b.get("id") for b in frame_boxes)

            for role, uid in targets:
                if uid is None:
                    continue

                if uid not in existing_ids:
                    name = self._object_name_for_id(
                        uid, default_for_none="", fallback=""
                    )
                    role_key = str(role or "").strip().lower()
                    time_key = str(time_label or "").strip().lower()

                    errors.append(
                        {
                            "msg": f"Frame {frame} ({time_label}) missing {role}: [{uid}] {name}",
                            "frame": frame,
                            "role": role,
                            "role_key": role_key,
                            "time_label": time_label,
                            "time_key": time_key,
                            "object_id": uid,
                            "object_name": name,
                            "slot": f"{role_key}_{time_key}",
                        }
                    )
        return errors

    def _add_anomaly_item(self, name: str, checked: bool = False):
        item = QListWidgetItem(name)
        item.setFlags(
            item.flags()
            | Qt.ItemIsUserCheckable
            | Qt.ItemIsEnabled
            | Qt.ItemIsEditable
        )
        item.setData(Qt.UserRole, name)
        item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self.anomaly_list.addItem(item)

    def _find_anomaly_item(self, name: str):
        for i in range(self.anomaly_list.count()):
            it = self.anomaly_list.item(i)
            if it.text().strip().lower() == name.strip().lower():
                return it
        return None

    def _set_anomaly_checked(self, name: str, checked: bool):
        it = self._find_anomaly_item(name)
        if not it:
            return
        it.setCheckState(Qt.Checked if checked else Qt.Unchecked)

    def _any_anomaly_checked(self) -> bool:
        for i in range(self.anomaly_list.count()):
            if self.anomaly_list.item(i).checkState() == Qt.Checked:
                return True
        return False

    def _apply_extra_label_config(self):
        """Update the UI (group box title and list items) from self.extra_label_config."""
        title = self.extra_label_config.get("title", "Hand Anomaly Label")
        self.group_anomaly.setTitle(title)

        self.anomaly_labels = list(self.extra_label_config.get("labels", []))
        self._init_anomaly_rules()

        self._anomaly_block = True
        try:
            self.anomaly_list.clear()
            default_lbl = self.extra_label_config.get("default_label", "Normal")
            for name in self.anomaly_labels:
                # Default to checked if it matches the default_label
                checked = (name.strip().lower() == default_lbl.strip().lower())
                self._add_anomaly_item(name, checked=checked)
        finally:
            self._anomaly_block = False

    def _selected_anomaly_label(self) -> str:
        """Return checked anomaly labels as a comma-separated string."""
        selected = []
        for i in range(self.anomaly_list.count()):
            it = self.anomaly_list.item(i)
            if it.checkState() == Qt.Checked:
                selected.append(it.text())
        if not selected:
            return self.extra_label_config.get("default_label", "Normal")
        fallback = self.extra_label_config.get("default_label", "Normal");
        if fallback in selected and len(selected) > 1:
            selected = [t for t in selected if t != fallback]
        return ", ".join(selected)

    def _set_selected_anomaly_label(self, label_str: str):
        """Parse stored label(s) and select all matching items."""
        target_keys = [t.strip().lower() for t in self._parse_anomaly_labels(label_str)]
        if not target_keys:
            fallback = self.extra_label_config.get("default_label", "Normal").lower();
            target_keys = [fallback]
        self._anomaly_block = True
        try:
            found_any = False
            for i in range(self.anomaly_list.count()):
                it = self.anomaly_list.item(i)
                txt = it.text().strip().lower()
                if txt in target_keys:
                    it.setCheckState(Qt.Checked)
                    found_any = True
                else:
                    it.setCheckState(Qt.Unchecked)
            if not found_any:
                self._set_anomaly_checked(self.extra_label_config.get("default_label", "Normal"), True)
        finally:
            self._anomaly_block = False

    def _on_anomaly_item_changed(self, item: QListWidgetItem):
        """
        Support multi-select anomaly labels with Normal as the exclusive fallback.
        """
        if self._anomaly_block:
            return

        old_text = (item.data(Qt.UserRole) or "").strip()
        curr_text = item.text().strip()
        if old_text and curr_text != old_text:
            if not curr_text:
                self._revert_anomaly_item_text(item, old_text)
                return
            existing = self._find_anomaly_item(curr_text)
            if existing and existing is not item:
                QMessageBox.information(self, "Info", f"Label '{curr_text}' already exists.")
                self._revert_anomaly_item_text(item, old_text)
                return
            self._rename_anomaly_item_logic(old_text, curr_text)
            self._log("hoi_anomaly_label_rename", old=old_text, new=curr_text)
            return

        self._anomaly_block = True
        try:
            is_checked = item.checkState() == Qt.Checked
            curr_text = item.text().strip()
            if is_checked:
                fallback = self.extra_label_config.get("default_label", "Normal").lower();
                if curr_text.lower() == fallback:
                    for i in range(self.anomaly_list.count()):
                        it = self.anomaly_list.item(i)
                        if it is not item:
                            it.setCheckState(Qt.Unchecked)
                else:
                    for i in range(self.anomaly_list.count()):
                        it = self.anomaly_list.item(i)
                        fallback = self.extra_label_config.get("default_label", "Normal").lower();
                        if it.text().strip().lower() == fallback:
                            it.setCheckState(Qt.Unchecked)

            if not self._any_anomaly_checked():
                self._set_anomaly_checked(self.extra_label_config.get("default_label", "Normal"), True)

            if (
                self.selected_hand_label
                and self.selected_hand_label in self.event_draft
            ):
                self.event_draft[self.selected_hand_label][
                    "anomaly_label"
                ] = self._selected_anomaly_label()
                self._update_status_label()
                if self.selected_event_id is not None:
                    self._apply_draft_to_selected_event()
                    self._refresh_events()
                    if getattr(self, "hoi_timeline", None):
                        self.hoi_timeline.refresh()

        finally:
            self._anomaly_block = False

    def _on_anomaly_title_edited(self, new_title: str):
        new_title = (new_title or "").strip()
        old_title = self.extra_label_config.get("title", "")
        if not new_title or new_title == old_title:
            self.group_anomaly.setTitle(old_title)
            return

        old_fallback = self.extra_label_config.get("default_label", "Normal")
        self.extra_label_config["title"] = new_title
        self.group_anomaly.setTitle(new_title)

        if old_title == "Hand Anomaly Label" and old_fallback == "Normal":
            self._rename_anomaly_item_logic("Normal", "Default")
            self.extra_label_config["default_label"] = "Default"
            self._apply_extra_label_config()

        self._log("hoi_anomaly_module_rename", title=new_title)

    def _on_anomaly_item_double_clicked(self, item: QListWidgetItem):
        if item is None:
            return
        self.anomaly_list.setCurrentItem(item)
        self.anomaly_list.editItem(item)

    def _revert_anomaly_item_text(self, item: QListWidgetItem, text: str):
        self._anomaly_block = True
        try:
            item.setText(text)
            item.setData(Qt.UserRole, text)
        finally:
            self._anomaly_block = False

    def _rename_anomaly_module(self):
        old_title = self.extra_label_config.get("title", "")
        old_fallback = self.extra_label_config.get("default_label", "Normal")
        new_title, ok = QInputDialog.getText(
            self, "Rename Module", "New title:", QLineEdit.Normal, old_title
        )
        if ok and new_title:
            self.extra_label_config["title"] = new_title
            self.group_anomaly.setTitle(new_title)
            
            # Auto-rename Normal -> Default if title is changed from default
            if old_title == "Hand Anomaly Label" and old_fallback == "Normal":
                self._rename_anomaly_item_logic("Normal", "Default")
                self.extra_label_config["default_label"] = "Default"
                self._apply_extra_label_config()

            self._log("hoi_anomaly_module_rename", title=new_title)

    def _rename_anomaly_item_logic(self, old_name: str, new_name: str):
        if "labels" in self.extra_label_config:
            for i, l in enumerate(self.extra_label_config["labels"]):
                if l.strip().lower() == old_name.strip().lower():
                    self.extra_label_config["labels"][i] = new_name
                    break
        if self.extra_label_config.get("default_label") == old_name:
            self.extra_label_config["default_label"] = new_name
            
        if "rules" in self.extra_label_config:
            if old_name in self.extra_label_config["rules"]:
                self.extra_label_config["rules"][new_name] = self.extra_label_config["rules"].pop(old_name)
        
        if hasattr(self, "anomaly_labels"):
            for i, l in enumerate(self.anomaly_labels):
                if l.strip().lower() == old_name.strip().lower():
                    self.anomaly_labels[i] = new_name
                    break
                    
        # Update event drafts (including the current draft)
        for actor in self.actors_config:
            hand_label = actor["id"]
            if hand_label in self.event_draft:
                cur = self.event_draft[hand_label].get("anomaly_label", "")
                if cur:
                    parts = [p.strip() for p in cur.split(",")]
                    replaced = False
                    for i, p in enumerate(parts):
                        if p.lower() == old_name.lower():
                            parts[i] = new_name
                            replaced = True
                    if replaced:
                        self.event_draft[hand_label]["anomaly_label"] = ", ".join(parts)
        
        # Update all events
        for ev in self.events:
            if "hoi_data" in ev:
                for actor in self.actors_config:
                    hand_key = actor["id"]
                    cur = ev["hoi_data"].get(hand_key, {}).get("anomaly_label", "")
                    if cur:
                        parts = [p.strip() for p in cur.split(",")]
                        replaced = False
                        for i, p in enumerate(parts):
                            if p.lower() == old_name.lower():
                                parts[i] = new_name
                                replaced = True
                        if replaced:
                            ev["hoi_data"][hand_key]["anomaly_label"] = ", ".join(parts)
        
        # Refresh UI
        self._apply_extra_label_config()
        if self.selected_hand_label and self.selected_hand_label in self.event_draft:
            current_labels = self.event_draft[self.selected_hand_label].get(
                "anomaly_label", ""
            )
            self._set_selected_anomaly_label(current_labels)
            self._update_status_label()
        if self.selected_event_id is not None:
            self._refresh_events()
            if getattr(self, "hoi_timeline", None):
                self.hoi_timeline.refresh()

    def _rename_anomaly_label(self):
        # ClickToggleList is in NoSelection mode; use currentItem or first checked item
        item = self.anomaly_list.currentItem()
        if not item:
            for i in range(self.anomaly_list.count()):
                it = self.anomaly_list.item(i)
                if it.checkState() == Qt.Checked:
                    item = it
                    break
        if not item:
            QMessageBox.information(self, "Info", "Please select or check a label to rename.")
            return

        old_name = item.text().strip()
        new_name, ok = QInputDialog.getText(
            self, "Rename Label", "New label name:", QLineEdit.Normal, old_name
        )
        if ok and new_name and new_name != old_name:
            self._rename_anomaly_item_logic(old_name, new_name)
            self._log("hoi_anomaly_label_rename", old=old_name, new=new_name)

    def _add_anomaly_label(self):
        name = self.anomaly_edit.text().strip()
        if not name:
            QMessageBox.information(self, "Info", "Please input a label name.")
            return
        if self._find_anomaly_item(name):
            QMessageBox.information(self, "Info", f"Label '{name}' already exists.")
            return
        self._add_anomaly_item(name, checked=False)
        self.anomaly_labels.append(name)
        if "labels" not in self.extra_label_config:
            self.extra_label_config["labels"] = []
        if name not in self.extra_label_config["labels"]:
            self.extra_label_config["labels"].append(name)
        self._ensure_anomaly_rules()
        self.anomaly_edit.clear()
        self._log("hoi_anomaly_add", name=name)

    def _remove_anomaly_label(self):
        checked_items = []
        for i in range(self.anomaly_list.count()):
            it = self.anomaly_list.item(i)
            if it.checkState() == Qt.Checked:
                checked_items.append(it)
        if not checked_items:
            return
        item = checked_items[0]
        name = item.text().strip().lower()
        default_lbl = self.extra_label_config.get("default_label", "Normal")
        if name == default_lbl.lower():
            QMessageBox.information(
                self, "Info", f"The '{default_lbl}' label cannot be removed."
            )
            return
        row = self.anomaly_list.row(item)
        self.anomaly_list.takeItem(row)
        for label in list(self.anomaly_labels):
            if label.strip().lower() == name:
                self.anomaly_labels.remove(label)
                if "labels" in self.extra_label_config:
                    for l in list(self.extra_label_config["labels"]):
                        if l.strip().lower() == name:
                            self.extra_label_config["labels"].remove(l)
                            break
                break
        self._ensure_anomaly_rules()
        if not self._any_anomaly_checked():
            self._set_selected_anomaly_label(self.extra_label_config.get("default_label", "Normal"))
        if self.selected_hand_label and self.selected_hand_label in self.event_draft:
            self.event_draft[self.selected_hand_label][
                "anomaly_label"
            ] = self._selected_anomaly_label()
            self._log(
                "hoi_anomaly_select",
                hand=self.selected_hand_label,
                label=self.event_draft[self.selected_hand_label]["anomaly_label"],
            )
        self._log("hoi_anomaly_remove", name=name)

    def _import_instruments(self):
        """Import instrument list from txt."""
        fp, _ = QFileDialog.getOpenFileName(
            self, "Import Instrument List", "", "Text Files (*.txt)"
        )
        if not fp:
            return
        self.combo_instrument.clear()
        self.combo_instrument.addItem("None", None)
        loaded = self._load_entities_into_combo(fp, self.combo_instrument)
        self._log("hoi_load_instruments", path=fp, count=loaded)

    def _import_targets(self):
        """Import target list from txt."""
        fp, _ = QFileDialog.getOpenFileName(
            self, "Import Target List", "", "Text Files (*.txt)"
        )
        if not fp:
            return
        self.combo_target.clear()
        self.combo_target.addItem("None", None)
        loaded = self._load_entities_into_combo(fp, self.combo_target)
        self._log("hoi_load_targets", path=fp, count=loaded)

    def _load_entities_into_combo(self, fp, combo):
        """
        Universal entity loader: reads TXT, assigns global IDs, populates combo box.
        """
        try:
            loaded_count = 0
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    category = line.strip()
                    if not category:
                        continue
                    
                    entity_name = category

                    if entity_name in self.global_object_map:
                        uid = self.global_object_map[entity_name]
                    else:
                        uid = self.object_id_counter
                        self.global_object_map[entity_name] = uid
                        self.object_id_counter += 1

                    self.id_to_category[entity_name] = category

                    display_text = f"[{uid}] {entity_name}"

                    if combo.findData(uid) == -1:
                        combo.addItem(display_text, uid)
                        loaded_count += 1

            QMessageBox.information(
                self,
                "Success",
                f"Loaded {loaded_count} entities to {combo.objectName() or 'list'}",
            )
            return loaded_count
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed: {e}")
            return 0

    def _on_object_list_menu(self, pos):
        """Triggered when right-clicking the Objects list."""
        item = self.list_objects.itemAt(pos)
        if not item:
            return

        box_data = item.data(Qt.UserRole)
        if not box_data:
            return

        if self._is_hand_label(box_data.get("label")):
            return

        menu = QMenu(self)

        action_change = menu.addAction("Change ID / Propagate...")
        action_change.triggered.connect(lambda: self._dialog_change_id(item))

        action_delete = menu.addAction("Delete Box")
        action_delete.triggered.connect(lambda: self._delete_box(item))

        menu.exec_(self.list_objects.mapToGlobal(pos))

    def _dialog_change_id(self, item):
        """[Fix] Dialog to change ID to ANY target (removes category restriction)."""
        box = item.data(Qt.UserRole)
        current_uid = box.get("id")

        curr_name = self._object_name_for_id(current_uid, fallback="Unknown")

        candidates = []
        for name, uid in self.global_object_map.items():
            candidates.append((name, uid))

        candidates.sort(key=lambda x: x[1])

        if not candidates:
            QMessageBox.warning(self, "Error", "Entity Library is empty.")
            return

        from PyQt5.QtWidgets import (
            QDialog,
            QVBoxLayout,
            QComboBox,
            QCheckBox,
            QDialogButtonBox,
        )

        d = QDialog(self)
        d.setWindowTitle(f"Change ID for {curr_name}")
        d.resize(400, 150)
        layout = QVBoxLayout(d)

        layout.addWidget(QLabel(f"Current: [{current_uid}] {curr_name}"))
        layout.addWidget(QLabel("Change to:"))

        combo = QComboBox()
        current_idx = 0
        for i, (name, uid) in enumerate(candidates):
            combo.addItem(f"[{uid}] {name}", uid)
            if uid == current_uid:
                current_idx = i
        combo.setCurrentIndex(current_idx)
        layout.addWidget(combo)

        chk_propagate = QCheckBox("Apply to subsequent frames (Auto-Track)")
        chk_propagate.setChecked(True)
        chk_propagate.setToolTip(
            "Use IoU overlap to apply changes to subsequent frames."
        )
        layout.addWidget(chk_propagate)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(d.accept)
        btns.rejected.connect(d.reject)
        layout.addWidget(btns)

        if d.exec_() == QDialog.Accepted:
            new_uid = combo.currentData()
            if new_uid == current_uid:
                return

            new_label = next(
                (n for n, u in self.global_object_map.items() if u == new_uid), ""
            )

            self._push_undo()

            self._update_raw_box_id(
                self.player.current_frame, box, current_uid, new_uid, new_label
            )

            box["id"] = new_uid
            box["label"] = new_label

            count = 1
            if chk_propagate.isChecked():
                count += self._propagate_id(box, new_uid, new_label)

            self._rebuild_bboxes_from_raw()
            self._refresh_boxes_for_frame(self.player.current_frame)

            self._log(
                "hoi_box_change_id",
                old_id=current_uid,
                new_id=new_uid,
                new_label=new_label,
                propagated=bool(chk_propagate.isChecked()),
                count=count,
            )
            QMessageBox.information(
                self,
                "Updated",
                f"Updated ID to [{new_uid}] {new_label}\nApplied to {count} frames.",
            )

    def _propagate_id(self, source_box, new_uid, new_label=None):
        """
        [Fix] Propagate ID change based on IoU only (ignoring category mismatch).
        Allows correcting misclassified objects across frames.
        """
        start_frame = source_box["frame"]

        curr_frame = start_frame
        curr_box_geo = [
            source_box["x1"],
            source_box["y1"],
            source_box["x2"],
            source_box["y2"],
        ]
        modified_count = 0

        while True:
            curr_frame += 1
            if curr_frame >= self.player.frame_count:
                break

            next_boxes = self.bboxes.get(curr_frame, [])
            if not next_boxes:
                break

            best_match = None
            max_iou = 0.0

            for nb in next_boxes:
                if self._is_hand_label(nb.get("label")):
                    continue

                def get_iou(boxA, boxB):
                    xA = max(boxA[0], boxB["x1"])
                    yA = max(boxA[1], boxB["y1"])
                    xB = min(boxA[2], boxB["x2"])
                    yB = min(boxA[3], boxB["y2"])
                    interArea = max(0, xB - xA) * max(0, yB - yA)
                    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
                    boxBArea = (boxB["x2"] - boxB["x1"]) * (boxB["y2"] - boxB["y1"])
                    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

                iou = get_iou(curr_box_geo, nb)

                if iou > 0.3:
                    if iou > max_iou:
                        max_iou = iou
                        best_match = nb

            if best_match:
                old_id = best_match.get("id")

                best_match["id"] = new_uid
                if new_label:
                    best_match["label"] = new_label

                self._update_raw_box_id(
                    curr_frame, best_match, old_id, new_uid, new_label
                )

                curr_box_geo = [
                    best_match["x1"],
                    best_match["y1"],
                    best_match["x2"],
                    best_match["y2"],
                ]
                modified_count += 1
            else:
                break

        return modified_count

    @staticmethod
    def _coords_close(a: dict, b: dict, tol: float = 1.0) -> bool:
        for key in ("x1", "y1", "x2", "y2"):
            if abs(float(a.get(key, 0.0)) - float(b.get(key, 0.0))) > tol:
                return False
        return True

    def _update_raw_box_id(
        self, frame: int, box: dict, old_id: int, new_id: int, new_label: str = None
    ):
        """[Fix] Update raw box ID and LABEL/CLASS to keep consistency."""
        target_frame = int(frame) - self.start_offset
        for rb in self.raw_boxes:
            if rb.get("orig_frame") != target_frame:
                continue
            if old_id is not None and rb.get("id") != old_id:
                continue
            if not self._coords_close(rb, box):
                continue

            rb["id"] = new_id
            if new_label:
                rb["label"] = new_label
                cid = self._class_id_for_label(new_label)
                if cid is not None:
                    rb["class_id"] = cid

    def _delete_box(self, item):
        """Delete specific box from the current frame."""
        box = item.data(Qt.UserRole)
        frame = self.player.current_frame

        if frame in self.bboxes and box in self.bboxes[frame]:
            self._push_undo()
            target_frame = int(frame) - self.start_offset
            self.raw_boxes = [
                rb
                for rb in self.raw_boxes
                if not (
                    rb.get("orig_frame") == target_frame
                    and rb.get("id") == box.get("id")
                    and rb.get("label") == box.get("label")
                    and self._coords_close(rb, box)
                )
            ]
            self._rebuild_bboxes_from_raw()
            self._refresh_boxes_for_frame(frame)
            self._log("hoi_box_delete", box_id=box.get("id"), frame=frame)

    def _save_hands_xml(self):
        """Export current hand BBoxes (including modifications) to CVAT XML format."""
        if not self.raw_boxes:
            QMessageBox.information(self, "Info", "No hand data to save.")
            return

        default_name = "hands_modified.xml"
        fp, _ = QFileDialog.getSaveFileName(
            self, "Save Hands XML", default_name, "XML Files (*.xml)"
        )
        if not fp:
            return

        try:
            import xml.etree.ElementTree as ET

            root = ET.Element("annotations")
            ET.SubElement(root, "meta")

            tracks = {actor["id"]: [] for actor in self.actors_config}

            for b in self.raw_boxes:
                lbl = b.get("label")
                if lbl in tracks:
                    tracks[lbl].append(b)

            for label, boxes in tracks.items():
                if not boxes:
                    continue

                boxes.sort(key=lambda x: x["orig_frame"])

                track_node = ET.SubElement(
                    root, "track", id=str(self.box_id_counter), label=label
                )
                self.box_id_counter += 1

                for b in boxes:
                    ET.SubElement(
                        track_node,
                        "box",
                        frame=str(b["orig_frame"]),
                        xtl=f"{b['x1']:.2f}",
                        ytl=f"{b['y1']:.2f}",
                        xbr=f"{b['x2']:.2f}",
                        ybr=f"{b['y2']:.2f}",
                        outside="0",
                        occluded="0",
                        keyframe="1",
                    )

            tree = ET.ElementTree(root)
            tree.write(fp, encoding="utf-8", xml_declaration=True)
            self._log("hoi_save_hands_xml", path=fp, count=len(self.raw_boxes))
            self._mark_hoi_saved()
            QMessageBox.information(self, "Saved", f"Hand XML saved to:\n{fp}")

        except Exception as ex:
            QMessageBox.critical(self, "Error", f"Failed to save XML:\n{ex}")

    def _reset_event_draft(self):
        """[Step 1] Initialize/reset event draft structure."""
        self.event_draft = {
            "verb": None,
            "instrument_object_id": None,
            "target_object_id": None,
        }
        for actor in self.actors_config:
            self.event_draft[actor["id"]] = self._blank_hand_data()
        if hasattr(self, "lbl_event_status"):
            self.lbl_event_status.setText("No event selected.")
        if hasattr(self, "anomaly_list"):
            self._set_selected_anomaly_label(self.extra_label_config.get("default_label", "Normal"))

    def _save_ui_to_hand_draft(self, hand_label: str):
        """[Restore] Save UI to draft using COMBO BOX."""
        if not hand_label or hand_label not in self.event_draft:
            return

        verb = self.combo_verb.currentText()

        inst_id = self.combo_instrument.currentData()
        target_id = self.combo_target.currentData()
        anomaly = self._selected_anomaly_label()

        hand_data = self.event_draft[hand_label]
        self._ensure_hand_annotation_state(hand_data)
        hand_data["verb"] = verb
        hand_data["instrument_object_id"] = inst_id
        hand_data["target_object_id"] = target_id
        hand_data["anomaly_label"] = anomaly

    def _rebuild_actor_checkboxes(self):
        # Clear old
        while self.actor_layout.count():
            item = self.actor_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
                item.widget().deleteLater()
        self.actor_controls.clear()

        # Re-add
        for actor in self.actors_config:
            aid = actor["id"]
            short = actor.get("short", aid[:1])
            chk = QCheckBox(short)
            chk.setToolTip(actor.get("label", aid))
            chk.toggled.connect(lambda on, a=aid: self._select_hand(a, on))
            self.actor_layout.addWidget(chk)
            self.actor_controls[aid] = chk

    def _on_configure_actors(self):
        """Show dialog to manage actors."""
        actions = ["Add Actor", "Remove Actor", "Rename Actor"]
        action, ok = QInputDialog.getItem(
            self, "Configure Actors", "Select action:", actions, 0, False
        )
        if not ok:
            return

        if action == "Add Actor":
            name, ok = QInputDialog.getText(self, "Add Actor", "Actor ID (e.g. hand_3):")
            if ok and name:
                self.actors_config.append(
                    {
                        "id": name,
                        "label": name.replace("_", " ").title(),
                        "short": name[:1].upper(),
                    }
                )
                self._reinit_actor_system()
        elif action == "Remove Actor":
            ids = [a["id"] for a in self.actors_config]
            name, ok = QInputDialog.getItem(
                self, "Remove Actor", "Select actor to remove:", ids, 0, False
            )
            if ok and len(self.actors_config) > 1:
                self.actors_config = [a for a in self.actors_config if a["id"] != name]
                self._reinit_actor_system()
        elif action == "Rename Actor":
            ids = [a["id"] for a in self.actors_config]
            old_name, ok = QInputDialog.getItem(
                self, "Rename Actor", "Select actor to rename:", ids, 0, False
            )
            if ok:
                new_name, ok = QInputDialog.getText(
                    self, "Rename Actor", f"New ID for {old_name}:", QLineEdit.Normal, old_name
                )
                if ok and new_name and new_name != old_name:
                    for a in self.actors_config:
                        if a["id"] == old_name:
                            a["id"] = new_name
                            a["label"] = new_name.replace("_", " ").title()
                            a["short"] = new_name[:1].upper()
                    self._reinit_actor_system()

    def _reinit_actor_system(self):
        """Update UI and Timeline after actor config change."""
        self._rebuild_actor_checkboxes()
        self._reset_event_draft()
        if hasattr(self, "hoi_timeline"):
            self.hoi_timeline.actors_config = self.actors_config
            self.hoi_timeline._reinit_rows()
            self.hoi_timeline.refresh()
        self._update_status_label()

    def _swap_frame_hands(self):
        """Swap boxes for the first two actors on current frame."""
        if len(self.actors_config) < 2:
            return

        a1 = self.actors_config[0]["id"]
        a2 = self.actors_config[1]["id"]

        if not self.player.cap:
            QMessageBox.information(
                self, "Info", "Load a video before swapping frame hands."
            )
            return
        frame_idx = int(self.player.current_frame)
        target_orig = frame_idx - int(self.start_offset)

        actor_ids = [a1, a2]
        hand_boxes = [
            rb
            for rb in self.raw_boxes
            if rb.get("orig_frame") == target_orig
            and rb.get("label") in actor_ids
        ]
        if not hand_boxes:
            QMessageBox.information(
                self, "Info", f"No boxes for {a1}/{a2} found on the current frame."
            )
            return

        self._push_undo()
        for rb in self.raw_boxes:
            if rb.get("orig_frame") != target_orig:
                continue
            lbl = rb.get("label")
            if lbl == a1:
                rb["label"] = a2
            elif lbl == a2:
                rb["label"] = a1

        self._rebuild_bboxes_from_raw()
        self._refresh_boxes_for_frame(frame_idx)
        self._log("hoi_frame_swap_hands", frame=frame_idx)
        QMessageBox.information(
            self, "Swapped", f"Swapped {a1}/{a2} boxes on this frame."
        )

    def _load_hand_draft_to_ui(self, hand_label: str):
        """[Restore] Load draft to UI using COMBO BOX."""
        if not hand_label or hand_label not in self.event_draft:
            return

        hand_data = self.event_draft[hand_label]
        self._ensure_hand_annotation_state(hand_data)

        self.combo_verb.blockSignals(True)
        self.combo_instrument.blockSignals(True)
        self.combo_target.blockSignals(True)

        try:
            verb = hand_data.get("verb", "")

            idx = self.combo_verb.findText(verb)
            if idx >= 0:
                self.combo_verb.setCurrentIndex(idx)
            else:
                self.combo_verb.setCurrentText(verb)

            inst_id = hand_data.get("instrument_object_id")
            idx_inst = self.combo_instrument.findData(inst_id)
            if idx_inst >= 0:
                self.combo_instrument.setCurrentIndex(idx_inst)
            else:
                self.combo_instrument.setCurrentIndex(0)

            target_id = hand_data.get("target_object_id")
            idx_tar = self.combo_target.findData(target_id)
            if idx_tar >= 0:
                self.combo_target.setCurrentIndex(idx_tar)
            else:
                self.combo_target.setCurrentIndex(0)

        finally:
            self.combo_verb.blockSignals(False)
            self.combo_instrument.blockSignals(False)
            self.combo_target.blockSignals(False)

        self._sync_action_panel_selection(hand_data.get("verb", ""))
        anomaly = hand_data.get("anomaly_label", self.extra_label_config.get("default_label", "Normal"))
        self._set_selected_anomaly_label(anomaly)

        self._update_status_label()

    def _update_status_label(self):
        """Refresh status bar showing the selected HandOI segment and hand details."""
        if not hasattr(self, "lbl_event_status"):
            return
        self._update_inspector_tab_labels()
        if self.selected_event_id is None:
            self.lbl_event_title.setText("No event selected")
            self.lbl_event_frames.setText("Frames: -")
            self.lbl_event_meta.setText("Verb: -   Instrument: -   Target: -")
            self.lbl_event_status.setText("No HandOI segment selected.")
            self._set_status_chip(getattr(self, "lbl_event_actor_chip", None), "Idle", "neutral")
            self._set_status_chip(getattr(self, "lbl_event_health_chip", None), "Waiting", "neutral")
            self._set_status_card_tone(getattr(self, "event_status_card", None), "neutral")
            for btn in (
                getattr(self, "btn_jump_start_chip", None),
                getattr(self, "btn_jump_onset_chip", None),
                getattr(self, "btn_jump_end_chip", None),
            ):
                if btn is not None:
                    btn.setEnabled(False)
            return

        hand = self.selected_hand_label
        if hand:
            h_data = self.event_draft.get(hand, {})
            self._ensure_hand_annotation_state(h_data)
            s = h_data.get("interaction_start")
            o = h_data.get("functional_contact_onset")
            e = h_data.get("interaction_end")
            s_txt = str(s) if s is not None else "-"
            o_txt = str(o) if o is not None else "-"
            e_txt = str(e) if e is not None else "-"
            anom = h_data.get("anomaly_label", "Normal")
            verb = h_data.get("verb") or "-"
            inst_id = h_data.get("instrument_object_id")
            target_id = h_data.get("target_object_id")
            inst_name = "-"
            target_name = "-"
            for name, id_val in self.global_object_map.items():
                if inst_id is not None and id_val == inst_id:
                    inst_name = name
                if target_id is not None and id_val == target_id:
                    target_name = name

            actor_label = self._get_actor_full_label(hand)
            missing = []
            if s is None or e is None:
                missing.append("start/end")
            if o is None:
                missing.append("onset")
            if not h_data.get("verb"):
                missing.append("verb")
            if inst_id is None:
                missing.append("instrument")
            if target_id is None:
                missing.append("target")
            evidence_summary = self._sparse_evidence_summary(h_data)
            evidence_expected = int(evidence_summary.get("expected", 0) or 0)
            evidence_confirmed = int(evidence_summary.get("confirmed", 0) or 0)
            evidence_missing = int(evidence_summary.get("missing", 0) or 0)
            if evidence_missing:
                missing.append(f"bbox {evidence_confirmed}/{evidence_expected}")
            suggested_fields = []
            for field_name in (
                "interaction_start",
                "functional_contact_onset",
                "interaction_end",
                "verb",
                "instrument_object_id",
                "target_object_id",
            ):
                state = get_field_state(h_data, field_name)
                if state.get("status") == "suggested":
                    suggested_fields.append(field_name)
            if missing:
                tone = "warn"
                health_text = f"Missing {len(missing)}"
            elif suggested_fields:
                tone = "warn"
                health_text = f"Review {len(suggested_fields)}"
            elif evidence_expected > 0:
                tone = "ok"
                health_text = f"Evidence {evidence_confirmed}/{evidence_expected}"
            else:
                tone = "ok"
                health_text = "Ready"

            self.lbl_event_title.setText(f"Event {self.selected_event_id}")
            self.lbl_event_frames.setText(f"Frames  Start {s_txt}   Onset {o_txt}   End {e_txt}")
            self.lbl_event_meta.setText(
                f"Verb: {verb}   Instrument: {inst_name}   Target: {target_name}   Anomaly: {anom}"
            )
            if missing:
                status_text = "Missing: " + ", ".join(missing)
            elif suggested_fields:
                labels = [
                    {
                        "interaction_start": "start",
                        "functional_contact_onset": "onset",
                        "interaction_end": "end",
                        "verb": "verb",
                        "instrument_object_id": "instrument",
                        "target_object_id": "target",
                    }.get(name, name)
                    for name in suggested_fields
                ]
                status_text = "Suggested, needs confirmation: " + ", ".join(labels)
            else:
                if evidence_expected > 0:
                    status_text = (
                        f"Complete event ready for review. Sparse evidence {evidence_confirmed}/{evidence_expected} grounded."
                    )
                else:
                    status_text = "Complete event ready for review."
            self.lbl_event_status.setText(status_text)
            self._set_status_chip(getattr(self, "lbl_event_actor_chip", None), actor_label, "neutral")
            self._set_status_chip(getattr(self, "lbl_event_health_chip", None), health_text, tone)
            self._set_status_card_tone(
                getattr(self, "event_status_card", None),
                "active" if not missing and not suggested_fields else "warn",
            )
            if getattr(self, "btn_jump_start_chip", None):
                self.btn_jump_start_chip.setEnabled(s is not None)
            if getattr(self, "btn_jump_onset_chip", None):
                self.btn_jump_onset_chip.setEnabled(o is not None)
            if getattr(self, "btn_jump_end_chip", None):
                self.btn_jump_end_chip.setEnabled(e is not None)
        else:
            self.lbl_event_title.setText(f"Event {self.selected_event_id}")
            self.lbl_event_frames.setText("Frames: -")
            self.lbl_event_meta.setText("Select an actor row to edit its metadata.")
            self.lbl_event_status.setText(f"Event {self.selected_event_id} selected.")
            self._set_status_chip(getattr(self, "lbl_event_actor_chip", None), "No actor", "warn")
            self._set_status_chip(getattr(self, "lbl_event_health_chip", None), "Pending", "warn")
            self._set_status_card_tone(getattr(self, "event_status_card", None), "warn")
            for btn in (
                getattr(self, "btn_jump_start_chip", None),
                getattr(self, "btn_jump_onset_chip", None),
                getattr(self, "btn_jump_end_chip", None),
            ):
                if btn is not None:
                    btn.setEnabled(False)
