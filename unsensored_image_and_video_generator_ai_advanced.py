#!/usr/bin/env python3
"""
Image Generator Pro - Расширенная версия с Hugging Face Diffusers
Используйте эту версию если хотите реальную генерацию изображений через AI
"""

import sys
import os
import subprocess
import threading
import traceback
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_CACHE_DIR = PROJECT_DIR / "models" / "huggingface"
MODEL_HUB_CACHE_DIR = MODEL_CACHE_DIR / "hub"
MODEL_HUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ['HF_HOME'] = str(MODEL_CACHE_DIR)
os.environ['HUGGINGFACE_HUB_CACHE'] = str(MODEL_HUB_CACHE_DIR)
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QLineEdit, QTextEdit, QPushButton, QComboBox, QCheckBox,
        QFrame, QMessageBox, QProgressBar, QFileDialog
    )
    from PyQt5.QtCore import Qt, pyqtSignal, QThread
    from PyQt5.QtGui import QPixmap, QFont
except ImportError:
    print("Install PyQt5: pip install PyQt5")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Install Pillow: pip install Pillow")
    sys.exit(1)


class SafetyFilter:
    """Проверка безопасности контента"""

    NSFW_KEYWORDS = {
        'nude', 'sex', 'porn', 'xxx', 'naked', 'explicit',
        'adult', 'erotic', 'sexual', 'arousal'
    }

    VIOLENCE_KEYWORDS = {
        'kill', 'murder', 'blood', 'violence', 'gun', 'bomb',
        'weapon', 'shoot', 'stab', 'hurt', 'harm', 'death'
    }

    HATE_KEYWORDS = {
        'hate', 'racism', 'racist', 'nazi', 'terrorist', 'extremist',
        'discrimination', 'bigot', 'offensive', 'slur'
    }

    ILLEGAL_KEYWORDS = {
        'bomb', 'explosive', 'drug', 'cocaine', 'heroin', 'meth',
        'stolen', 'counterfeit', 'fake', 'illegal'
    }

    @staticmethod
    def check(text, filters):
        """Проверка текста на соответствие фильтрам"""
        lower_text = text.lower()
        words = set(lower_text.split())

        if filters.get('nsfw', True):
            if SafetyFilter.NSFW_KEYWORDS & words:
                return False, "NSFW content is blocked"

        if filters.get('violence', True):
            if SafetyFilter.VIOLENCE_KEYWORDS & words:
                return False, "Violent content is blocked"

        if filters.get('hate', True):
            if SafetyFilter.HATE_KEYWORDS & words:
                return False, "Hateful content is blocked"

        if filters.get('illegal', True):
            if SafetyFilter.ILLEGAL_KEYWORDS & words:
                return False, "Illegal content is blocked"

        return True, "OK"


class HuggingFaceImageWorker(QThread):
    """Генерация изображений с Hugging Face"""

    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, prompt, style, quality):
        super().__init__()
        self.prompt = prompt
        self.style = style
        self.quality = quality
        self.pipe = None

    def run(self):
        """Генерация изображения"""
        try:
            self.progress.emit("Starting isolated generator process...")
            script = PROJECT_DIR / "generate_image_cli.py"
            command = [
                sys.executable,
                "-u",
                str(script),
                "--prompt",
                self.prompt,
                "--quality",
                self.quality,
            ]
            env = os.environ.copy()
            env["HF_HOME"] = str(MODEL_CACHE_DIR)
            env["HUGGINGFACE_HUB_CACHE"] = str(MODEL_HUB_CACHE_DIR)
            env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_DIR),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
            )

            image_path = None
            output_lines = []
            assert process.stdout is not None
            for line in process.stdout:
                message = line.rstrip()
                if not message:
                    continue
                output_lines.append(message)
                print(message, flush=True)
                if message.startswith("RESULT_PATH="):
                    image_path = message.split("=", 1)[1]
                elif not message.startswith(("Loading pipeline components", "  0%", "100%|")):
                    self.progress.emit(message)

            return_code = process.wait()
            if return_code != 0:
                details = "\n".join(output_lines[-8:])
                self.error.emit(
                    f"Generator process exited with code {return_code}.\n{details}"
                )
                return

            if not image_path or not Path(image_path).exists():
                self.error.emit("Generator finished but did not report a saved image path.")
                return

            self.finished.emit(image_path)
            return

            self.progress.emit("Loading AI model. First run downloads several GB; keep the app open.")

            if DIFFUSERS_IMPORT_ERROR is not None:
                self.error.emit(
                    "Failed to load Diffusers/Torch. "
                    f"{type(DIFFUSERS_IMPORT_ERROR).__name__}: {DIFFUSERS_IMPORT_ERROR}"
                )
                return

            # Выбираем модель в зависимости от качества
            model_name = {
                'standard': os.getenv('SD_MODEL_STANDARD', 'stable-diffusion-v1-5/stable-diffusion-v1-5'),
                'hd': os.getenv('SD_MODEL_HD', 'stable-diffusion-v1-5/stable-diffusion-v1-5'),
                '4k': os.getenv('SD_MODEL_4K', 'stabilityai/stable-diffusion-2-1')
            }.get(self.quality, os.getenv('SD_MODEL_STANDARD', 'stable-diffusion-v1-5/stable-diffusion-v1-5'))

            # Используем CPU или GPU
            force_cpu = os.getenv('SD_DEVICE', '').lower() == 'cpu'
            device = 'cuda' if torch.cuda.is_available() and not force_cpu else 'cpu'
            vram_gb = 0
            if device == 'cuda':
                props = torch.cuda.get_device_properties(0)
                vram_gb = props.total_memory / 1024 ** 3
                self.progress.emit(f"Device: CUDA - {props.name} ({vram_gb:.1f} GB VRAM)")
            else:
                self.progress.emit("Device: CPU")

            # Загружаем модель
            self.progress.emit(f"Model cache: {MODEL_HUB_CACHE_DIR}")
            self.progress.emit(f"Loading model: {model_name}")
            model_variant = os.getenv('SD_MODEL_VARIANT', 'fp16')
            if model_variant and model_variant.lower() in {'none', 'default', 'false', '0'}:
                model_variant = None
            self.progress.emit(f"Model variant: {model_variant or 'default'}")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
                use_safetensors=True,
                variant=model_variant,
                cache_dir=str(MODEL_HUB_CACHE_DIR),
                safety_checker=None  # Используем свои фильтры
            )
            self.pipe.enable_attention_slicing()
            if device == 'cuda' and vram_gb and vram_gb < 6:
                self.progress.emit("Low-VRAM GPU detected; using CPU offload to avoid CUDA memory errors.")
                self.pipe.enable_model_cpu_offload()
            else:
                self.pipe = self.pipe.to(device)

            # Размер изображения
            size = {
                'standard': 512,
                'hd': 768,
                '4k': 768
            }.get(self.quality, 512)
            if device == 'cuda' and vram_gb and vram_gb < 6 and size > 512:
                self.progress.emit("Low-VRAM GPU detected; using 512x512 output for this run.")
                size = 512

            steps = int(os.getenv('SD_STEPS', '25'))
            guidance_scale = float(os.getenv('SD_GUIDANCE_SCALE', '7.5'))
            self.progress.emit(f"Generating image at {size}x{size} for {steps} steps...")

            # Генерируем
            with torch.no_grad():
                latents = self.pipe(
                    self.prompt,
                    height=size,
                    width=size,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    output_type='latent'
                ).images[0]

            self.progress.emit("Decoding image on CPU to avoid CUDA VAE crashes.")
            vae = AutoencoderKL.from_pretrained(
                model_name,
                subfolder='vae',
                torch_dtype=torch.float32,
                use_safetensors=True,
                cache_dir=str(MODEL_HUB_CACHE_DIR)
            ).to('cpu')
            vae.enable_slicing()

            latents = latents.detach().to('cpu', dtype=torch.float32).unsqueeze(0)
            scaling_factor = getattr(vae.config, 'scaling_factor', 0.18215)
            with torch.no_grad():
                decoded = vae.decode(latents / scaling_factor, return_dict=False)[0]
            image = self.pipe.image_processor.postprocess(decoded, output_type='pil')[0]
            del vae
            if device == 'cuda':
                torch.cuda.empty_cache()

            extrema = image.convert('RGB').getextrema()
            if all(low == high for low, high in extrema):
                raise RuntimeError(
                    "The model returned a blank image. This usually means invalid fp16 output on this GPU. "
                    "Try setting SD_MODEL_VARIANT=none or run on CPU."
                )

            # Сохраняем
            output_dir = PROJECT_DIR / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = output_dir / f"image_ai_{timestamp}.png"
            image.save(filename)

            self.progress.emit(f"Image saved: {filename}")
            print(f"Image saved: {filename}", flush=True)
            self.finished.emit(str(filename))

        except Exception as e:
            traceback.print_exc()
            self.error.emit(f"Error: {str(e)}")


class AdvancedImageGeneratorApp(QMainWindow):
    """Приложение с поддержкой Hugging Face"""

    def __init__(self):
        super().__init__()
        self.worker = None
        self.current_image = None
        self.init_ui()

    def init_ui(self):
        """Инициализация интерфейса"""
        self.setWindowTitle("Image Generator Pro (AI Version)")
        self.setGeometry(100, 100, 1200, 700)
        self.setStyleSheet(self._get_stylesheet())

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        left_panel = self._create_left_panel()
        right_panel = self._create_right_panel()

        main_layout.addWidget(left_panel, 1)
        main_layout.addWidget(right_panel, 1)

    def _create_left_panel(self):
        """Левая панель с контролами"""
        frame = QFrame()
        layout = QVBoxLayout(frame)
        layout.setSpacing(10)

        title = QLabel("AI Settings")
        title.setFont(QFont('Arial', 14, QFont.Bold))
        layout.addWidget(title)

        safety_info = QFrame()
        safety_layout = QVBoxLayout(safety_info)
        safety_msg = QLabel("Real AI image generation with Hugging Face Diffusers")
        safety_msg.setWordWrap(True)
        safety_msg.setStyleSheet("color: #93c5fd; background: rgba(59, 130, 246, 0.1); padding: 10px; border-radius: 5px;")
        safety_layout.addWidget(safety_msg)
        layout.addWidget(safety_info)

        layout.addWidget(QLabel("Prompt (English):"))
        self.prompt_input = QTextEdit()
        self.prompt_input.setPlaceholderText("Describe the image you want to generate...")
        self.prompt_input.setMaximumHeight(120)
        layout.addWidget(self.prompt_input)

        layout.addWidget(QLabel("Quality:"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["standard", "hd", "4k"])
        layout.addWidget(self.quality_combo)

        layout.addWidget(QLabel("Safety filters:"))

        self.filters = {}
        for name, label in [
            ('nsfw', 'Block NSFW content'),
            ('violence', 'Block violence'),
            ('hate', 'Block hateful content'),
            ('illegal', 'Block illegal content')
        ]:
            self.filters[name] = QCheckBox(label)
            self.filters[name].setChecked(True)
            layout.addWidget(self.filters[name])

        layout.addSpacing(20)

        self.generate_btn = QPushButton("Generate with AI")
        self.generate_btn.setFont(QFont('Arial', 11, QFont.Bold))
        self.generate_btn.setMinimumHeight(45)
        self.generate_btn.clicked.connect(self.generate_image)
        layout.addWidget(self.generate_btn)

        self.download_btn = QPushButton("Save image")
        self.download_btn.setMinimumHeight(40)
        self.download_btn.clicked.connect(self.download_image)
        self.download_btn.setEnabled(False)
        layout.addWidget(self.download_btn)

        layout.addStretch()

        info = QLabel("v2.0 AI - Hugging Face Diffusers")
        info.setStyleSheet("color: #64748b; font-size: 11px;")
        layout.addWidget(info)

        return frame

    def _create_right_panel(self):
        """Правая панель с превью"""
        frame = QFrame()
        layout = QVBoxLayout(frame)
        layout.setSpacing(10)

        title = QLabel("Result")
        title.setFont(QFont('Arial', 14, QFont.Bold))
        layout.addWidget(title)

        self.preview_label = QLabel()
        self.preview_label.setMinimumSize(400, 400)
        self.preview_label.setStyleSheet(
            "border: 2px dashed #475569; border-radius: 10px; "
            "background: rgba(15, 23, 42, 0.6);"
        )
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setText("Your AI image will appear here")
        layout.addWidget(self.preview_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #cbd5e1; font-size: 12px;")
        layout.addWidget(self.status_label)

        return frame

    def generate_image(self):
        """Генерация изображения"""
        prompt = self.prompt_input.toPlainText().strip()

        if not prompt:
            self.show_error("Please enter an English prompt")
            return

        filters = {k: v.isChecked() for k, v in self.filters.items()}
        safe, reason = SafetyFilter.check(prompt, filters)

        if not safe:
            self.show_error(reason)
            return

        self.generate_btn.setEnabled(False)
        self.progress_bar.setVisible(True)

        self.worker = HuggingFaceImageWorker(
            prompt,
            'realistic',
            self.quality_combo.currentText()
        )
        self.worker.finished.connect(self.on_generation_finished)
        self.worker.error.connect(self.on_generation_error)
        self.worker.progress.connect(self.update_status)
        self.worker.start()

    def on_generation_finished(self, image_path):
        """Завершение генерации"""
        self.current_image = image_path
        self.progress_bar.setValue(100)

        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaledToWidth(400, Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled_pixmap)

        self.download_btn.setEnabled(True)
        self.generate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

        self.show_success("AI image generated successfully!")

    def on_generation_error(self, error_msg):
        """Обработка ошибки"""
        self.generate_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.show_error(error_msg)

    def update_status(self, message):
        """Обновление статуса"""
        self.status_label.setText(message)

    def download_image(self):
        """Скачивание изображения"""
        if not self.current_image:
            return

        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getSaveFileName(
            self, "Save image", "", "PNG (*.png);;JPG (*.jpg)"
        )

        if file_path:
            import shutil
            shutil.copy(self.current_image, file_path)
            self.show_success(f"Image saved: {file_path}")

    def show_error(self, message):
        """Сообщение об ошибке"""
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color: #fca5a5; font-size: 12px;")

    def show_success(self, message):
        """Успешное сообщение"""
        self.status_label.setText(message)
        self.status_label.setStyleSheet("color: #bbf7d0; font-size: 12px;")

    def _get_stylesheet(self):
        """Стили приложения"""
        return """
            QMainWindow { background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); }
            QFrame { background: rgba(30, 41, 59, 0.8); border: 1px solid rgba(148, 163, 184, 0.2);
                     border-radius: 10px; padding: 15px; }
            QLabel { color: #e2e8f0; }
            QTextEdit, QComboBox { background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(148, 163, 184, 0.3);
                                   border-radius: 6px; color: #e2e8f0; padding: 8px; }
            QTextEdit:focus, QComboBox:focus { border: 1px solid #06b6d4; background: rgba(15, 23, 42, 0.9); }
            QPushButton { background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%); color: white;
                         border: none; border-radius: 6px; font-weight: bold; padding: 10px; }
            QPushButton:hover { background: linear-gradient(135deg, #0891b2 0%, #2563eb 100%); }
            QPushButton:disabled { background: #64748b; color: #94a3b8; }
            QCheckBox { color: #cbd5e1; }
            QProgressBar { background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(148, 163, 184, 0.3);
                          border-radius: 6px; }
            QProgressBar::chunk { background: linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%); }
        """


def main():
    app = QApplication(sys.argv)
    window = AdvancedImageGeneratorApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
