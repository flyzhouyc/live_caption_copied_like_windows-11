import sys
import json
import os
import queue
import numpy as np
import vosk
import sounddevice as sd
from datetime import datetime
from contextlib import ExitStack

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSettings, QCoreApplication
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QTextEdit,
    QPushButton,
    QComboBox,
    QLabel,
    QFileDialog,
    QStatusBar,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
    QDialog,
    QFormLayout,
    QSpinBox,
    QDialogButtonBox,
    QFontComboBox,
    QLineEdit,
    QCheckBox,
    QColorDialog,
    QListWidget,
    QListWidgetItem
)
from PySide6.QtGui import QTextCursor, QIcon, QColor, QFont

# Use the external pyqtdarktheme package.
import qdarktheme

# Import resampling tool from SciPy.
from scipy.signal import resample

MODEL_DIR = "models"
CAPTURE_DIR = "captures"

###############################################################################
# Helper function to determine device type based on its name.
###############################################################################
def get_device_type(device_name):
    name = device_name.lower()
    if "built-in" in name or "internal" in name:
        return "Internal"
    elif ("headphone" in name or "headset" in name or "usb" in name or "earphone" in name):  # <-- Added 'earphone' here
        return "External"
    else:
        return "Unknown"

###############################################################################
# SpeechThread for a single device (supports optional loopback)
###############################################################################
class SpeechThread(QThread):
    update_text = Signal(str, bool)  # text, is_partial
    language_detected = Signal(str)

    def __init__(self, model_path, sample_rate=16000, block_size=8000, device=None, loopback=False):
        super().__init__()
        # If a device is specified, query the appropriate info.
        if device is not None:
            try:
                info = sd.query_devices(device, 'input' if not loopback else 'output')
                self.sample_rate = info.get("default_samplerate", sample_rate)
            except Exception:
                self.sample_rate = sample_rate
        else:
            self.sample_rate = sample_rate
        self.block_size = block_size
        self.device = device
        self.loopback = loopback
        self.model = vosk.Model(model_path)
        self.recognizer = vosk.KaldiRecognizer(self.model, self.sample_rate)
        self.running = False
        self.q = queue.Queue()
        self.current_language = self.detect_language(model_path)

    def detect_language(self, model_path):
        dir_name = os.path.basename(model_path)
        parts = dir_name.split("-")
        if len(parts) >= 3:
            lang_code = parts[2]
        else:
            lang_code = "unknown"
        return lang_code.upper()

    def run(self):
        self.running = True
        try:
            kwargs = {
                "samplerate": self.sample_rate,
                "blocksize": self.block_size,
                "dtype": 'int16',
                "channels": 1,
                "device": self.device,
                "callback": self.audio_callback
            }

            with sd.RawInputStream(**kwargs):
                self.language_detected.emit(self.current_language)
                while self.running:
                    data = self.q.get()
                    if self.recognizer.AcceptWaveform(data):
                        result = json.loads(self.recognizer.Result())
                        self.update_text.emit(result.get('text', ''), False)
                    else:
                        partial = json.loads(self.recognizer.PartialResult())
                        self.update_text.emit(partial.get('partial', ''), True)
        except Exception as e:
            self.update_text.emit(f"Error: {str(e)}", False)

    def audio_callback(self, indata, frames, time, status):
        self.q.put(bytes(indata))

    def stop(self):
        self.running = False

###############################################################################
# MultiSpeechThread for multiple devices (each with its own config)
# device_configs is a list of tuples: (device_index, loopback_flag)
###############################################################################
class MultiSpeechThread(QThread):
    update_text = Signal(str, bool)
    language_detected = Signal(str)

    def __init__(self, model_path, device_configs, target_sample_rate=16000, block_size=8000):
        super().__init__()
        self.device_configs = device_configs  # e.g. [(3, False), (5, True)]
        self.target_sample_rate = target_sample_rate
        self.block_size = block_size  # target block size (in samples at target rate)
        self.model = vosk.Model(model_path)
        self.recognizer = vosk.KaldiRecognizer(self.model, self.target_sample_rate)
        self.running = False
        self.queues = {}
        self.device_rates = {}
        for (dev, loopback) in self.device_configs:
            self.queues[(dev, loopback)] = queue.Queue()
            try:
                info = sd.query_devices(dev, 'input' if not loopback else 'output')
                self.device_rates[(dev, loopback)] = info.get("default_samplerate", self.target_sample_rate)
            except Exception:
                self.device_rates[(dev, loopback)] = self.target_sample_rate
        self.current_language = "UNKNOWN"

    def make_callback(self, dev, loopback):
        def callback(indata, frames, time, status):
            self.queues[(dev, loopback)].put(bytes(indata))
        return callback

    def run(self):
        self.running = True
        with ExitStack() as stack:
            streams = []
            for (dev, loopback) in self.device_configs:
                device_rate = self.device_rates[(dev, loopback)]
                # Adjust block size so that capture duration remains consistent.
                device_block_size = int(self.block_size * device_rate / self.target_sample_rate)
                cb = self.make_callback(dev, loopback)


                stream = stack.enter_context(
                    sd.RawInputStream(
                            samplerate=device_rate,
                            blocksize=device_block_size,
                            dtype='int16',
                            channels=1,
                            device=dev,
                            callback=cb
                    )
                )
                streams.append(stream)

            self.language_detected.emit(self.current_language)
            while self.running:
                data_list = []
                for (dev, loopback) in self.device_configs:
                    try:
                        data = self.queues[(dev, loopback)].get(timeout=1.0)
                        arr = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                        device_rate = self.device_rates[(dev, loopback)]
                        if device_rate != self.target_sample_rate:
                            arr = resample(arr, self.block_size)
                        data_list.append(arr)
                    except queue.Empty:
                        data_list.append(np.zeros(self.block_size, dtype=np.float32))

                combined = np.mean(data_list, axis=0)
                combined_int16 = np.clip(combined, -32768, 32767).astype(np.int16)
                combined_bytes = combined_int16.tobytes()

                if self.recognizer.AcceptWaveform(combined_bytes):
                    result = json.loads(self.recognizer.Result())
                    self.update_text.emit(result.get('text', ''), False)
                else:
                    partial = json.loads(self.recognizer.PartialResult())
                    self.update_text.emit(partial.get('partial', ''), True)

    def stop(self):
        self.running = False

###############################################################################
# Settings Dialog with separate Input and Output device selection,
# Always on Top and Auto Switch options.
###############################################################################
class SettingsDialog(QDialog):
    def __init__(self, current_settings, default_settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(500, 600)
        self.default_settings = default_settings.copy()
        self.current_settings = current_settings.copy()
        self.font_color = self.current_settings.get("font_color", "#dcdcdc")

        layout = QFormLayout(self)

        # Target Sample Rate
        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setRange(8000, 48000)
        self.sample_rate_spin.setValue(self.current_settings.get("sample_rate", 16000))
        layout.addRow("Target Sample Rate (Hz):", self.sample_rate_spin)

        # Block Size
        self.block_size_spin = QSpinBox()
        self.block_size_spin.setRange(1000, 16000)
        self.block_size_spin.setValue(self.current_settings.get("block_size", 8000))
        layout.addRow("Block Size (samples):", self.block_size_spin)

        # Theme selection (Dark or Light)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        self.theme_combo.setCurrentText(self.current_settings.get("theme", "Dark"))
        layout.addRow("Theme:", self.theme_combo)

        # Font Family selection
        self.font_combo = QFontComboBox()
        self.font_combo.setCurrentFont(QFont(self.current_settings.get("font_family", "Segoe UI")))
        layout.addRow("Font Family:", self.font_combo)

        # Font Size
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(6, 72)
        self.font_size_spin.setValue(self.current_settings.get("font_size", 15))
        layout.addRow("Font Size:", self.font_size_spin)

        # Font Color
        self.font_color_button = QPushButton("Select Font Color")
        self.font_color_button.setStyleSheet(f"background-color: {self.font_color};")
        self.font_color_button.clicked.connect(self.choose_font_color)
        layout.addRow("Font Color:", self.font_color_button)

        # Window Transparency
        self.transparency_spin = QSpinBox()
        self.transparency_spin.setRange(0, 100)
        self.transparency_spin.setValue(self.current_settings.get("transparency", 100))
        layout.addRow("Window Transparency (%):", self.transparency_spin)

        # Text Transparency Checkbox
        self.text_transparency_checkbox = QCheckBox("Allow text transparency")
        self.text_transparency_checkbox.setChecked(self.current_settings.get("text_transparency", False))
        layout.addRow("Text Transparency:", self.text_transparency_checkbox)

        # Always on Top Checkbox
        self.always_on_top_checkbox = QCheckBox("Always on Top")
        self.always_on_top_checkbox.setChecked(self.current_settings.get("always_on_top", False))
        layout.addRow("Always on Top:", self.always_on_top_checkbox)

        # Auto Switch Headphone Checkbox
        self.auto_switch_checkbox = QCheckBox("Auto Switch to Headphone")
        self.auto_switch_checkbox.setChecked(self.current_settings.get("auto_switch_headphone", False))
        layout.addRow("Auto Switch Headphone:", self.auto_switch_checkbox)

        # Separate lists for Input and Output audio devices.
        self.input_list_widget = QListWidget()
        self.input_list_widget.setSelectionMode(QListWidget.MultiSelection)
        self.output_list_widget = QListWidget()
        self.output_list_widget.setSelectionMode(QListWidget.MultiSelection)
        all_devices = sd.query_devices()
        for i, d in enumerate(all_devices):
            # For input devices.
            if d.get("max_input_channels", 0) > 0:
                host_api_info = sd.query_hostapis(d["hostapi"])
                host_api_name = host_api_info.get("name", "Unknown API")
                default_sr = d.get("default_samplerate", "unknown")
                item_text = f"{d['name']} - {host_api_name}, {default_sr} Hz (ID: {i}) [Input]"
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, i)
                self.input_list_widget.addItem(item)
            # For output devices.
            if d.get("max_output_channels", 0) > 0:
                host_api_info = sd.query_hostapis(d["hostapi"])
                host_api_name = host_api_info.get("name", "Unknown API")
                default_sr = d.get("default_samplerate", "unknown")
                item_text = f"{d['name']} - {host_api_name}, {default_sr} Hz (ID: {i}) [Output]"
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, i)
                self.output_list_widget.addItem(item)

        layout.addRow("Input Audio Devices:", self.input_list_widget)
        layout.addRow("Output Audio Devices:", self.output_list_widget)

        # Buttons: OK, Cancel, Restore Defaults.
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.restore_button = QPushButton("Restore Defaults")
        layout.addRow(self.restore_button)
        layout.addWidget(self.buttons)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.restore_button.clicked.connect(self.restore_defaults)

    def choose_font_color(self):
        color = QColorDialog.getColor(QColor(self.font_color), self, "Select Font Color")
        if color.isValid():
            self.font_color = color.name()
            self.font_color_button.setStyleSheet(f"background-color: {self.font_color};")

    def restore_defaults(self):
        self.sample_rate_spin.setValue(self.default_settings.get("sample_rate", 16000))
        self.block_size_spin.setValue(self.default_settings.get("block_size", 8000))
        self.theme_combo.setCurrentText(self.default_settings.get("theme", "Dark"))
        self.font_combo.setCurrentFont(QFont(self.default_settings.get("font_family", "Segoe UI")))
        self.font_size_spin.setValue(self.default_settings.get("font_size", 15))
        self.font_color = self.default_settings.get("font_color", "#dcdcdc")
        self.font_color_button.setStyleSheet(f"background-color: {self.font_color};")
        self.transparency_spin.setValue(self.default_settings.get("transparency", 100))
        self.text_transparency_checkbox.setChecked(self.default_settings.get("text_transparency", False))
        self.always_on_top_checkbox.setChecked(self.default_settings.get("always_on_top", False))
        self.auto_switch_checkbox.setChecked(self.default_settings.get("auto_switch_headphone", False))
        self.input_list_widget.clearSelection()
        self.output_list_widget.clearSelection()

    def get_settings(self):
        input_selected = []
        for item in self.input_list_widget.selectedItems():
            input_selected.append(str(item.data(Qt.UserRole)))

        output_selected = []
        for item in self.output_list_widget.selectedItems():
            output_selected.append(str(item.data(Qt.UserRole)))

        return {
            "sample_rate": self.sample_rate_spin.value(),
            "block_size": self.block_size_spin.value(),
            "theme": self.theme_combo.currentText(),
            "font_family": self.font_combo.currentFont().family(),
            "font_size": self.font_size_spin.value(),
            "font_color": self.font_color,
            "transparency": self.transparency_spin.value(),
            "text_transparency": self.text_transparency_checkbox.isChecked(),
            "always_on_top": self.always_on_top_checkbox.isChecked(),
            "auto_switch_headphone": self.auto_switch_checkbox.isChecked(),
            "audio_input_devices": ",".join(input_selected),
            "audio_output_devices": ",".join(output_selected)
        }

###############################################################################
# Main Application with Persistent Settings and Multi-Device Support
###############################################################################
class LiveCaptionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.speech_thread = None
        self.start_time = None
        self.current_save_filename = None
        self.saved_lines = set()

        # Extend default settings with new keys.
        self.default_settings = {
            "sample_rate": 16000,
            "block_size": 8000,
            "theme": "Dark",
            "font_family": "Segoe UI",
            "font_size": 15,
            "font_color": "#dcdcdc",
            "transparency": 100,
            "text_transparency": False,
            "always_on_top": False,
            "auto_switch_headphone": False,
            "audio_input_devices": "",
            "audio_output_devices": ""
        }
        self.settings = self.load_settings()

        try:
            qdarktheme.setup_theme(self.settings["theme"].lower())
        except AttributeError:
            app = QApplication.instance()
            if app:
                app.setStyleSheet(qdarktheme.load_stylesheet(self.settings["theme"].lower()))

        self.init_ui()
        self.apply_custom_settings()

        self.full_text = []
        self.available_models = []
        self.load_models()

    def load_settings(self):
        s = QSettings("MyCompany", "LiveCaptionPro")
        loaded = {}
        loaded["sample_rate"] = int(s.value("sample_rate", self.default_settings["sample_rate"]))
        loaded["block_size"] = int(s.value("block_size", self.default_settings["block_size"]))
        loaded["theme"] = s.value("theme", self.default_settings["theme"])
        loaded["font_family"] = s.value("font_family", self.default_settings["font_family"])
        loaded["font_size"] = int(s.value("font_size", self.default_settings["font_size"]))
        loaded["font_color"] = s.value("font_color", self.default_settings["font_color"])
        loaded["transparency"] = int(s.value("transparency", self.default_settings["transparency"]))
        loaded["text_transparency"] = s.value("text_transparency", "false").lower() == "true"
        loaded["always_on_top"] = s.value("always_on_top", str(self.default_settings["always_on_top"])).lower() == "true"
        loaded["auto_switch_headphone"] = s.value("auto_switch_headphone", str(self.default_settings["auto_switch_headphone"])).lower() == "true"
        loaded["audio_input_devices"] = s.value("audio_input_devices", self.default_settings["audio_input_devices"])
        loaded["audio_output_devices"] = s.value("audio_output_devices", self.default_settings["audio_output_devices"])
        return loaded

    def save_settings(self):
        s = QSettings("MyCompany", "LiveCaptionPro")
        s.setValue("sample_rate", self.settings["sample_rate"])
        s.setValue("block_size", self.settings["block_size"])
        s.setValue("theme", self.settings["theme"])
        s.setValue("font_family", self.settings["font_family"])
        s.setValue("font_size", self.settings["font_size"])
        s.setValue("font_color", self.settings["font_color"])
        s.setValue("transparency", self.settings["transparency"])
        s.setValue("text_transparency", self.settings["text_transparency"])
        s.setValue("always_on_top", self.settings["always_on_top"])
        s.setValue("auto_switch_headphone", self.settings["auto_switch_headphone"])
        s.setValue("audio_input_devices", self.settings["audio_input_devices"])
        s.setValue("audio_output_devices", self.settings["audio_output_devices"])

    def init_ui(self):
        self.setWindowTitle("Live Caption Pro")
        self.setGeometry(100, 100, 1000, 700)
        self.setWindowIcon(QIcon("icon.png"))

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        menubar = self.menuBar()
        settings_menu = menubar.addMenu("Settings")
        settings_action = settings_menu.addAction("Preferences")
        settings_action.triggered.connect(self.open_settings_dialog)

        model_layout = QHBoxLayout()
        self.model_selector = QComboBox()
        self.lbl_language = QLabel("Detected Language: ")
        model_layout.addWidget(QLabel("Select Model:"))
        model_layout.addWidget(self.model_selector, 2)
        model_layout.addWidget(self.lbl_language, 1)
        layout.addLayout(model_layout)

        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        layout.addWidget(self.text_display)

        button_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start", objectName="startButton")
        self.btn_stop = QPushButton("Stop", objectName="stopButton")
        self.btn_save = QPushButton("Save Now", objectName="saveButton")
        self.btn_stop.setEnabled(False)
        button_layout.addWidget(self.btn_start)
        button_layout.addWidget(self.btn_stop)
        button_layout.addWidget(self.btn_save)
        layout.addLayout(button_layout)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.btn_start.clicked.connect(self.start_recording)
        self.btn_stop.clicked.connect(self.stop_recording)
        self.btn_save.clicked.connect(self.manual_save)

    def hex_to_rgba(self, hex_color, transparency):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6:
            return hex_color
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        a = int(transparency / 100 * 255)
        return f"rgba({r}, {g}, {b}, {a})"

    def apply_custom_settings(self):
        theme_choice = self.settings.get("theme", "Dark").lower()
        try:
            qdarktheme.setup_theme(theme_choice)
        except AttributeError:
            app = QApplication.instance()
            if app:
                app.setStyleSheet(qdarktheme.load_stylesheet(theme_choice))

        font = QFont(self.settings.get("font_family", "Segoe UI"), self.settings.get("font_size", 15))
        self.text_display.setFont(font)

        if self.settings.get("text_transparency", False):
            color = self.hex_to_rgba(self.settings.get("font_color", "#dcdcdc"), self.settings.get("transparency", 100))
        else:
            color = self.settings.get("font_color", "#dcdcdc")

        self.text_display.setStyleSheet(f"color: {color};")
        self.btn_start.setStyleSheet(f"color: {color};")
        self.btn_stop.setStyleSheet(f"color: {color};")
        self.btn_save.setStyleSheet(f"color: {color};")

        self.setWindowOpacity(self.settings.get("transparency", 100) / 100.0)

        # Apply "Always on Top" setting.
        if self.settings.get("always_on_top", False):
            self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        else:
            self.setWindowFlag(Qt.WindowStaysOnTopHint, False)
        self.show()  # Refresh window flags

    def open_settings_dialog(self):
        dialog = SettingsDialog(self.settings, self.default_settings, parent=self)
        if dialog.exec() == QDialog.Accepted:
            new_settings = dialog.get_settings()
            self.settings.update(new_settings)
            self.apply_custom_settings()
            self.save_settings()
            self.status_bar.showMessage("Settings updated", 3000)

    def load_models(self):
        if os.path.exists(MODEL_DIR):
            self.available_models = [
                d for d in os.listdir(MODEL_DIR)
                if os.path.isdir(os.path.join(MODEL_DIR, d))
            ]
        else:
            self.available_models = []
        if not self.available_models:
            QMessageBox.critical(self, "Error", "No models found in the models directory")
            return
        self.model_selector.addItems(self.available_models)

    def start_recording(self):
        if not self.available_models:
            return

        if self.current_save_filename is None:
            if not os.path.exists(CAPTURE_DIR):
                os.makedirs(CAPTURE_DIR)
            lang_code = (
                self.lbl_language.text().split()[-1].lower()
                if self.lbl_language.text() else "unknown"
            )
            self.current_save_filename = os.path.join(
                CAPTURE_DIR, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{lang_code}.txt"
            )

        model_name = self.model_selector.currentText()
        model_path = os.path.join(MODEL_DIR, model_name)
#设置
        # Build device configurations from manually selected input and output devices.
        device_configs = []
        # If Auto Switch Headphone is enabled, override manual selection.
        if self.settings.get("auto_switch_headphone", False):
            all_devices = sd.query_devices()
            for i, d in enumerate(all_devices):
                name_lower = d["name"].lower()
                if "headphone" in name_lower or "headset" in name_lower or "earphone" in name_lower:  # <-- Added 'earphone' here
                    # Always add microphone input if available.
                    if d.get("max_input_channels", 0) > 0:
                        device_configs.append((i, False))
         
        else:
            input_str = self.settings.get("audio_input_devices", "")

            if input_str:
                for id_str in input_str.split(","):
                    try:
                        dev_id = int(id_str.strip())
                        device_configs.append((dev_id, False))
                    except ValueError:
                        pass
     

        if device_configs:
            try:
                # If more than one device configuration is selected, use the multi-device thread.
                if len(device_configs) > 1:
                    self.speech_thread = MultiSpeechThread(
                        model_path,
                        device_configs,
                        target_sample_rate=self.settings["sample_rate"],
                        block_size=self.settings["block_size"]
                    )
                else:
                    # Single device: use SpeechThread.
                    dev, loopback = device_configs[0]
                    self.speech_thread = SpeechThread(
                        model_path,
                        sample_rate=self.settings["sample_rate"],
                        block_size=self.settings["block_size"],
                        device=dev,
                        loopback=loopback
                    )

                self.speech_thread.update_text.connect(self.update_caption)
                self.speech_thread.language_detected.connect(self.update_language)
                self.speech_thread.start()
                self.start_time = datetime.now()
                self.btn_start.setEnabled(False)
                self.btn_stop.setEnabled(True)
                self.model_selector.setEnabled(False)
                self.status_bar.showMessage("Recording...")
                self.auto_save_timer = QTimer()
                self.auto_save_timer.timeout.connect(self.auto_save)
                self.auto_save_timer.start(30000)
            except Exception as e:
                self.status_bar.showMessage(f"Error: {str(e)}")
        else:
            self.status_bar.showMessage("No valid audio devices selected.")

    def stop_recording(self):
        if self.speech_thread:
            self.speech_thread.stop()
            self.speech_thread.quit()
            self.speech_thread.wait()
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.model_selector.setEnabled(True)
            self.status_bar.showMessage("Ready")
            self.auto_save_timer.stop()
            self.auto_save()

    def update_caption(self, text, is_partial):
        cursor = self.text_display.textCursor()
        cursor.movePosition(QTextCursor.End)

        if is_partial:
            if hasattr(self, 'partial_start'):
                cursor.setPosition(self.partial_start)
                cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
                cursor.removeSelectedText()
            else:
                self.partial_start = cursor.position()
            self.text_display.setTextColor(QColor('#888'))
            cursor.insertText(text)
        else:
            if hasattr(self, 'partial_start'):
                cursor.setPosition(self.partial_start)
                cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
                cursor.removeSelectedText()
                del self.partial_start
            text = text.strip()
            if text and not text.endswith(('.', '!', '?')):
                text += ". "
            self.text_display.setTextColor(QColor('#dcdcdc'))
            cursor.insertText("\n" + text)
            self.full_text.append(text)
        self.text_display.moveCursor(QTextCursor.End)

    def update_language(self, lang_code):
        self.lbl_language.setText(f"Detected Language: {lang_code}")

    def auto_save(self):
        if self.current_save_filename is None:
            return
        new_lines = []
        for line in self.full_text[-10:]:
            if line not in self.saved_lines:
                new_lines.append(line)
                self.saved_lines.add(line)
        if new_lines:
            try:
                with open(self.current_save_filename, 'a', encoding='utf-8') as f:
                    f.write("\n".join(new_lines) + "\n")
                self.status_bar.showMessage(f"Auto-saved to {self.current_save_filename}", 3000)
                self.full_text = []
            except Exception as e:
                self.status_bar.showMessage(f"Auto-save error: {str(e)}", 3000)

    def manual_save(self):
        options = QFileDialog.Options()
        default_name = self.current_save_filename if self.current_save_filename else "captions.txt"
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Captions", default_name, "Text Files (*.txt)", options=options
        )
        if filename:
            new_lines = []
            for line in self.full_text:
                if line not in self.saved_lines:
                    new_lines.append(line)
                    self.saved_lines.add(line)
            try:
                with open(filename, 'a', encoding='utf-8') as f:
                    f.write("\n".join(new_lines) + "\n")
                self.status_bar.showMessage(f"Saved to {filename}", 5000)
                self.full_text = []
            except Exception as e:
                self.status_bar.showMessage(f"Save error: {str(e)}", 5000)

###############################################################################
# Main Entry Point
###############################################################################
if __name__ == "__main__":
    QCoreApplication.setOrganizationName("MyCompany")
    QCoreApplication.setApplicationName("LiveCaptionPro")
    app = QApplication(sys.argv)
    window = LiveCaptionApp()
    window.show()
    window.status_bar.showMessage("Load models from: " + os.path.abspath(MODEL_DIR))
    sys.exit(app.exec())
