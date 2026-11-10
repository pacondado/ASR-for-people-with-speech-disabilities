# Whisper LoRA Fine-tuning

**Teach speech recognition to understand your speech. Do it yourself, on your own computer, in your own language.**

Author: **[Paulo Condado](https://paulocondado.net)** · License: **MIT** (free to use, change and share)

---

## Why this project

Dictation, automatic subtitles and voice assistants work well for "typical" speech, but make
many mistakes when speech is different: people with speech disorders (such as dysarthria),
strong accents or a very specific vocabulary.

Instead of waiting for big companies to fix this, you can do it yourself.

## What you will do

1. **Record** sentences spoken by the person whose speech you want the computer to understand.
2. **Write down** what was said in each recording, in a spreadsheet.
3. **Train**: one command, and the computer learns that speech.
4. **Test**: compare your model with the original and see how much it improved.

It uses **Whisper**, a free speech-recognition model by OpenAI, and adapts it to your speech with
a technique called **LoRA**. You don't need to be a programmer: if you can copy and paste, you
can do it. Your recordings never leave your computer.

## Who it is for

- People with speech disorders, their families, carers and speech therapists.
- Makers and curious people who like to build their own tools.
- Researchers working on atypical, accented or regional speech.

## Why do it yourself

This project follows a **constructionist approach**: people learn best, and get tools that really
fit them, when they **build things themselves** instead of only receiving ready-made solutions.

The author is a researcher with cerebral palsy. Speech-recognition products are designed for the
"average" voice, and people with atypical speech are rarely the ones who decide how these tools
work. Here, the people who need the technology — and those around them — can build it, test it
and improve it for their own speech, in their own language.

## What you need

- A Mac with an Apple chip (M1 or later) or a computer with an NVIDIA graphics card.
- Speech recordings (a few hundred short sentences) and their transcriptions.
- An internet connection for the installation.

---

## Contents

| Step | What you do | Time |
|---|---|---|
| [Before you start](#before-you-start) | Check your computer and learn to use the terminal. | 10 min |
| [Step 1: Install](#step-1-install) | Copy and paste a few commands. | 15–30 min, once |
| [Step 2: Record](#step-2-record) | Record sentences of the speech to learn. | A few sessions |
| [Step 3: List your recordings](#step-3-list-your-recordings) | Write what was said in each recording, in a spreadsheet. | 1–2 hours |
| [Step 4: Train](#step-4-train) | Run one command and wait. | 1–3 hours |
| [Step 5: Test](#step-5-test) | See how much better your model is. | 15–30 min |
| [Step 6: Use your model](#step-6-use-your-model) | Transcribe new recordings. | 5 min |

Also: [Why do it yourself](#why-do-it-yourself) · [Make it yours](#make-it-yours) · [Words used in this guide](#words-used-in-this-guide) ·
[Something went wrong?](#something-went-wrong) · [Reference for advanced users](#reference-for-advanced-users) ·
[About the author](#about-the-author) · [Research](#research) · [License](#license)

---

## Before you start

Check that you have:

- **A suitable computer.** A Mac with an Apple chip (M1, M2, M3, M4…), or a computer with an
  NVIDIA graphics card. Not sure? On a Mac, open the Apple menu → *About This Mac*: if it says "Chip Apple M…",
  you're good. Developed and tested on Apple Silicon Macs
- **An internet connection** for the installation and the first training (to download Whisper).
- **A microphone** (a phone or a headset is fine) and a quiet room.

### Opening the terminal

All the steps use the **terminal**: a window where you type (or paste) commands and press
**Enter**.

| System | How to open it |
|---|---|
| macOS | Press `Cmd` + `Space`, type **Terminal**, press Enter. |
| Windows | Open the Start menu, type **Miniforge Prompt** (available after step 1.1). |
| Linux | Press `Ctrl` + `Alt` + `T`. |

Tips:

- You can **paste** in the terminal with `Cmd` + `V` (Mac) or right-click (Windows / Linux).
- Lines starting with `#` in the examples are comments: they explain, and are ignored.
- If a command is still running, wait until you see the prompt (the line where you type) again.

---

## Step 1: Install

### 1.1 Install Miniforge

Miniforge creates an isolated "box" with the right version of Python, so nothing else on your
computer is affected.

Download and install it from **[github.com/conda-forge/miniforge](https://github.com/conda-forge/miniforge#install)**
(follow the instructions for your system), then **close and reopen the terminal**.

### 1.2 Download this project

Click the green **Code** button at the top of this page → **Download ZIP**, and unzip it, for
example into your *Documents* folder. Then, in the terminal, go to that folder:

```bash
cd ~/Documents/whisper-lora-finetuning
```

(Adjust the folder name if yours is different. Tip: type `cd ` with a space and drag the folder
into the terminal window.)

### 1.3 Install everything

Copy and paste, one block at a time:

```bash
# Create the box and enter it
conda create -n whisper-lora python=3.10 -y
conda activate whisper-lora
```

```bash
# Install the tools (this takes a few minutes)
conda install -c conda-forge ffmpeg -y
pip install -r src/train/whisper_lora_finetune/requirements.txt
pip install torchcodec epitran
```

**Done.** Each time you open a new terminal to use this project, first run:

```bash
cd ~/Documents/whisper-lora-finetuning
conda activate whisper-lora
```

---

## Step 2: Record

This is the most important step: **the model can only learn from what you give it.**

**How many recordings?** Start with **300 to 800 sentences**. The author's experiments used
about 750. More recordings usually give better results.

**How to record:**

- **One sentence per file**, between 2 and 20 seconds.
- A **quiet room**, always the same microphone if possible.
- Speak **naturally**, as you normally do. Don't try to speak "better": the goal is for the
  model to learn the real speech.
- Use sentences from **everyday life**: names of family and friends, places, things you ask for,
  words from your work or hobbies.
- Save as **WAV** files (most recording apps can do this; [Audacity](https://www.audacityteam.org/)
  is free).
- Give the files **simple names**: `rec_0001.wav`, `rec_0002.wav`, …

**Recording from the terminal (Linux).** You can also record directly with `ffmpeg`, which was
installed in step 1:

```bash
ffmpeg -f alsa -i default -ac 1 -ar 16000 -c:a pcm_s16le rec_0001.wav
```

Start speaking after you press Enter, and press `q` to stop the recording. Then repeat with the
next name (`rec_0002.wav`, …), or `t0001.wav`, `t0002.wav`, … for the test recordings.

| Part of the command | What it means |
|---|---|
| `-f alsa -i default` | Record from the computer's default microphone. |
| `-ac 1` | One channel (mono). |
| `-ar 16000` | 16 000 samples per second, the rate Whisper uses. |
| `-c:a pcm_s16le` | Standard uncompressed WAV (16-bit). |
| `rec_0001.wav` | Name of the file to create. |

If you get `Unknown input format: 'alsa'`, install the system version of ffmpeg
(`sudo apt install ffmpeg` on Ubuntu/Debian) and run the command again.

Put all the recordings in the folder `data/train/audio/` inside the project.

> **Keep some recordings apart for testing.** Record an extra 50 to 100 different sentences and
> don't use them for training: in [step 5](#step-5-test) they show whether the model really
> learned, or just memorised.

---

## Step 3: List your recordings

The computer needs to know **what was said in each recording**. You write this in a spreadsheet
with two columns:

| audio | text |
|---|---|
| rec_0001.wav | Bom dia. Como estás hoje? |
| rec_0002.wav | Queria um copo de água, por favor. |
| rec_0003.wav | O comboio para Lisboa parte às nove. |

- The first row must be exactly **`audio`** and **`text`** (lowercase).
- **audio**: the file name, exactly as it is in the folder (with `.wav`).
- **text**: exactly what was said, with normal spelling and punctuation.

Save it as **`data.csv`** in the folder `data/train/`:

| Program | How to save |
|---|---|
| Google Sheets | *File → Download → Comma-separated values (.csv)* |
| LibreOffice Calc | *File → Save As → Text CSV*, character set **UTF-8**, field delimiter **`,`** (comma) |
| Excel | *File → Save As → CSV UTF-8 (Comma delimited)* |

> **Check the separator.** In some countries (Portugal, Brazil, Spain, France, …) Excel saves CSV
> files with `;` instead of `,`. Open `data.csv` with a text editor: the lines must look like
> `rec_0001.wav,Bom dia. Como estás hoje?` with a **comma** after the file name.

### Or write it in a simple text editor

A CSV file is just text, so you can also write it by hand in any basic text editor. Type it like
this:

```
audio,text
rec_0001.wav,Bom dia. Como estás hoje?
rec_0002.wav,"Queria um copo de água, por favor."
rec_0003.wav,"Ele foi jantar e, depois das 21 horas, regressou."
rec_0004.wav,O comboio para Lisboa parte às nove.
```

The rules:

1. The **first line** is always `audio,text`.
2. **One recording per line**: the file name, a **comma**, and what was said.
3. **No spaces** before or after that first comma.
4. If the text **contains a comma**, put the whole text between double quotes: `"…, …"`.
5. If the text **contains double quotes**, write each one twice and put the whole text between
   quotes, as in the line `rec_0003.wav` above.

How to save it as `data.csv` in the folder `data/train/`:

| System | Editor | How to save |
|---|---|---|
| macOS | **TextEdit** | First choose *Format → Make Plain Text*. Then *File → Save*, name it `data.csv`, choose encoding **Unicode (UTF-8)**, and if asked, choose **Use .csv**. |
| Windows | **Notepad** | *File → Save as*, *Save as type*: **All files (\*.\*)**, name it `data.csv`, *Encoding*: **UTF-8**. |
| Linux | **Text Editor** (gedit) or any other | *Save As*, name it `data.csv`. UTF-8 is already the default. |

Make sure the file is called `data.csv` and not `data.csv.txt`: on Windows, turn on
*View → Show → File name extensions* in the File Explorer (*View → File name extensions* on
Windows 10) to see the full name.

The test file `test.csv` (step 5) is written in exactly the same way.

Your folders should now look like this:

```
data/
└── train/
    ├── data.csv
    └── audio/
        ├── rec_0001.wav
        ├── rec_0002.wav
        └── ...
```

---

## Step 4: Train

### 4.1 Choose your language

Find your language code. **Use the same code in every command.**

| Language | Code | | Language | Code |
|---|---|---|---|---|
| Portuguese | `pt` | | Italian | `it` |
| English | `en` | | Dutch | `nl` |
| Spanish | `es` | | Catalan | `ca` |
| French | `fr` | | Polish | `pl` |
| German | `de` | | Greek | `el` |

Whisper supports about 100 languages; for others, search "Whisper language codes".

### 4.2 Start training

Paste this command (change `pt` to your language code if needed):

```bash
python src/train/whisper_lora_finetune/train.py \
    --dataset data/train/data.csv \
    --audio-dir data/train/audio/ \
    --model medium \
    --language pt
```

> **Windows:** put the whole command on one line, without the `\` characters.

**What you'll see:** first some information about your data, then a progress bar like
`8%|▊ | 40/520 [04:21<52:07]` — that means 40 of 520 steps done, about 52 minutes to go.
You can leave the computer working; on a Mac, keep it plugged in and awake.

**How long?** About **1 hour** for ~765 sentences with `medium` on an Apple Silicon Mac.

**The first time**, Whisper is downloaded (about 3 GB). If some recordings are longer than
30 seconds, a larger Whisper model is also downloaded to cut them into pieces.

### 4.3 Where is my model?

When it finishes, a new folder appears with a name like:

```
output_lora_medium_r16_a32_lr6e-05_640_160_20261109-153000
```

That name describes your training: model `medium`, settings `r16`, `a32`, `lr6e-05`,
640 sentences used for learning, 160 for checking, and the date and time.

Inside, the folder **`lora_weights`** is **your model** — the part that learned your speech. It's
small: a few tens of megabytes, while Whisper itself is about 3 GB.

### 4.4 Want better results? The author's recommended settings

This is the command used in the author's experiments, with a bigger model (`large-v3`, more
accurate but about 2–3× slower) and fine-tuned settings:

```bash
python src/train/whisper_lora_finetune/train.py \
    --dataset data/train/data.csv \
    --audio-dir data/train/audio/ \
    --model large-v3 \
    --language pt \
    --val-ratio 0.15 \
    --seed 2026 \
    --epochs 13 \
    --group-by-text \
    --early-stopping-patience 4 \
    --min-val-words 4 \
    --long-audio keep \
    --long-audio-train-only \
    --learning-rate 4e-5 \
    --lora-r 8 \
    --lora-alpha 16
```

What each setting does is explained in the [reference for advanced users](#reference-for-advanced-users), at the end.

---

## Step 5: Test

### 5.1 Prepare the test recordings

Use the extra recordings you kept apart in step 2 — **never the ones used for training**.
Put them in a folder **with the name of the person**, and list them in `test.csv` the same way
as in step 3:

```
data/
└── test/
    ├── test.csv
    └── audio/
        └── Maria/
            ├── t0001.wav
            ├── t0002.wav
            └── ...
```

`test.csv`:

```csv
audio,text
t0001.wav,Ontem esteve imenso calor.
t0002.wav,"Amanhã de manhã, vou limpar os vidros."
```

(When the text has a comma, the spreadsheet adds the quotes `"…"` for you.)

You can test several people: create one folder per person (`audio/Maria/`, `audio/João/`, …),
each with recordings of the **same sentences** listed in `test.csv`.

### 5.2 Run the test

Replace the folder name with yours from step 4.3, and use the **same model size** (`medium` or
`large-v3`) and **language** as in training:

```bash
python src/evaluation/evaluation/evaluate.py \
    --base-model medium \
    --model output_lora_medium_r16_a32_lr6e-05_640_160_20261109-153000/lora_weights \
    --data data/test \
    --language pt \
    --normalize
```

The test compares two models on every test sentence (the example below is simplified):

- **BASE**: the original Whisper, before learning your speech;
- **LoRA**: your model.

### 5.3 Read the results

For each sentence you'll see what was really said (**REF**), what each model wrote, and its
scores:

```
REF:
  Amanhã de manhã vou aspirar a casa e limpar os vidros.

BASE:
  A manhã de manhã vou espirar a casa e limpar os vídeos.
  WER=0.364  CER=0.075

LoRA:
  Amanhã de manhã vou aspirar a casa e limpar os vidros.
  WER=0.000  CER=0.000
```

The main number is the **WER (word error rate)**: the share of words that are wrong.

| WER | Meaning |
|---|---|
| 0.00 | Perfect: every word right. |
| 0.10 | 1 word in 10 is wrong. |
| 0.30 | 3 words in 10 are wrong. |

**Lower is better.** At the end you get a summary per person: if your model's WER is lower than
BASE, it learned. The full results are saved in the folder `output_whisper/`, in a file with the
name of your model.

The other numbers are explained in [Words used in this guide](#words-used-in-this-guide).

---

## Step 6: Use your model

Save this as `transcribe.py` in the project folder, change the three lines in capitals, and run
`python transcribe.py`:

```python
import torch
import librosa
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL_FOLDER = "output_lora_medium_r16_a32_lr6e-05_640_160_20261109-153000"   # your model
WHISPER_SIZE = "medium"                                                      # same as in training
LANGUAGE = "pt"                                                              # same as in training

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
processor = WhisperProcessor.from_pretrained(f"{MODEL_FOLDER}/processor")
base = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{WHISPER_SIZE}")
model = PeftModel.from_pretrained(base, f"{MODEL_FOLDER}/lora_weights").to(device).eval()

audio, _ = librosa.load("my_recording.wav", sr=16000, mono=True)
inputs = processor(audio, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)

with torch.no_grad():
    ids = model.generate(
        input_features=inputs.input_features.to(device),
        attention_mask=inputs.attention_mask.to(device),
        language=LANGUAGE,
        task="transcribe",
    )

print(processor.batch_decode(ids, skip_special_tokens=True)[0])
```

It prints the text of `my_recording.wav` (any recording up to 30 seconds).

---

## Make it yours

This project is meant to be **taken apart, changed and improved**. Some ideas:

**Get better results**

| Try | How | Why |
|---|---|---|
| More recordings | Record more sentences and train again. | The most effective improvement of all. |
| A bigger model | `--model large-v3` (and `--base-model large-v3` in the test). | More accurate, but slower. |
| Other settings | Change `--lora-r`, `--lora-alpha` (keep it 2 × r) and `--learning-rate`. | Each person's speech is different. |
| Many settings at once | Edit the lists at the top of `run_lora.sh` and run `bash run_lora.sh`. | Trains every combination while you sleep. |

**Compare two models** and see whether one is really better or just lucky:

```bash
python src/evaluation/evaluation/compare_results.py output_whisper/MODEL_A.json output_whisper/MODEL_B.json --normalize
```

> **Good to know:** training twice with exactly the same settings gives slightly different
> results (it starts from random values). Don't choose settings from a single try: repeat the
> best ones with `--seed 1`, `--seed 2`, … and compare.

**Ideas for projects:** dictation for someone who can't type easily, subtitles for a family
video, voice commands for a home-made assistant, a speech-therapy progress diary, support for a
regional language or dialect…

**Share what you learn:** open an *Issue* on this page to ask questions, report problems or tell
us what worked for your speech and language. Improvements are welcome as *Pull Requests*.

> **Respect people's privacy.** Speech recordings are personal data. Only record people who agreed, and only
> share recordings — or models trained on them — with their consent.

---

## Words used in this guide

| Word | Meaning |
|---|---|
| **Whisper** | A free speech-recognition model by OpenAI: it turns speech into text. |
| **Fine-tuning** | Teaching an existing model something new (here: a particular person's speech). |
| **LoRA** | A way of fine-tuning that only adds a small extra part to Whisper, so it works on a normal computer. |
| **Training** | The process where the model learns from your recordings. |
| **Validation** | Part of your recordings set aside during training to check progress. |
| **Test** | Separate recordings, never used in training, to measure the final result. |
| **Epoch** | One pass through all the training recordings. |
| **WER** | Word error rate: share of words wrong (0.10 = 1 in 10). Lower is better. |
| **CER** | Character error rate: the same, counted letter by letter. Useful for small spelling mistakes. |
| **PER** | Phoneme error rate: the same, counted sound by sound. Shows when the words are wrong but sound right. |
| **Terminal** | The window where you type commands. |
| **CSV** | A simple table file, which spreadsheets can save. |

---

## Something went wrong?

| Message or problem | What to do |
|---|---|
| `conda: command not found` | Close and reopen the terminal after installing Miniforge. |
| `No such file or directory` | You're not in the project folder. Run the `cd …` command from step 1.2. |
| `Missing audio file: …` | A name in `data.csv` doesn't match a file in the folder. Check spelling and `.wav`. |
| `Column 'audio' not found` | The first row must be `audio,text`. Check the separator is a comma (step 3). |
| Accents look strange (`Ã©` instead of `é`) | Save the CSV as **UTF-8** (step 3). |
| Training is very slow | Look for the `Device` line at the start: it should say `mps` (Mac) or `cuda` (NVIDIA). `cpu` means the graphics chip isn't being used. |
| Out of memory | Add `--batch-size 1 --grad-accum 16` to the training command, or use a smaller `--model`. |
| The test fails to load your model | `--base-model` must be the same size used in training. |
| Your model is worse than BASE | Usually too few or too varied recordings. Record more, and check the transcriptions match the audio. |

Still stuck? Open an *Issue* on this page with the message you see and the command you ran.

---

## Reference for advanced users

### Training options

Run `python src/train/whisper_lora_finetune/train.py --help` for the complete list.

| Option | Default | What it does |
|---|---|---|
| `--dataset` | — | CSV with `audio` and `text` columns; split automatically into training and validation. |
| `--train-csv` / `--val-csv` | — | Use your own validation file instead of `--dataset`. |
| `--audio-dir` | `audio` | Folder with the audio files. |
| `--model` | `medium` | `small`, `medium` or `large-v3`. |
| `--language` | `pt` | Language code. |
| `--val-ratio` | `0.2` | Share of the data used for validation. |
| `--seed` | `42` | Makes the split and training reproducible. |
| `--epochs` | `13` | Maximum passes over the data. |
| `--early-stopping-patience` | `2` | Stop after N epochs without improvement. |
| `--group-by-text` | off | The same sentence never appears in both training and validation. |
| `--min-val-words` | `2` | Validation only uses sentences with at least N words. |
| `--max-samples` | all | Use at most N sentences in total. |
| `--long-audio` | `segment` | Recordings over 30 s: `segment` (cut into aligned pieces), `keep` (as is) or `drop`. |
| `--long-audio-train-only` | off | Long recordings are only used for training. |
| `--learning-rate` | `6e-5` | Learning speed. |
| `--lora-r` / `--lora-alpha` | `16` / `32` | Size and strength of the adapter (keep alpha = 2 × r). |
| `--batch-size` / `--grad-accum` | `2` / `8` | Lower `--batch-size` if you run out of memory. |

The adapter saved in `lora_weights/` comes from the epoch with the lowest validation WER.
Each training folder also contains `run_config.json` (all settings), `train_split.csv` and
`validation_split.csv` (which sentences were used where) and `validation_results.json`
(result per validation sentence).

To understand why two trainings differ:
`python src/train/whisper_lora_finetune/diagnose_runs.py FOLDER_A FOLDER_B`

### Test options and other languages

Run `python src/evaluation/evaluation/evaluate.py --help` for the complete list.

| Option | What it does |
|---|---|
| `--base-model` | `small`, `medium` or `large-v3` — same as in training. |
| `--model` | The `lora_weights` folder of your model. |
| `--data` | Folder with `test.csv` and `audio/<person>/`. |
| `--test-csv` | Use a test CSV located elsewhere. |
| `--language` | Same language code as in training. |
| `--normalize` | Ignore upper/lower case and punctuation when counting errors. |
| `--speakers` | Test only some people, e.g. `--speakers Maria`. |
| `--output` | Name of the results file (default: `output_whisper/<model folder>.json`). |
| `--no-per` | Skip the phoneme error rate. |

**Descriptions of the people** (shown in the report, and usable with
`compare_results.py --description`): edit `PARTICIPANTS` in
`src/evaluation/evaluation/whisper_eval/config.py`. The default list contains the author's
example speakers; people without a folder are skipped.

**Languages other than Portuguese:** WER and CER work for any language. For languages written
without spaces (Chinese, Japanese, Thai) look at the CER rather than the WER. The phoneme error
rate (PER) uses European Portuguese rules by default; for other languages set `EPITRAN_CODE` in
`whisper_eval/config.py` (e.g. `"spa-Latn"`, `"fra-Latn"`, `"deu-Latn"`, `"ita-Latn"`) and add
`--g2p epitran`, or skip it with `--no-per`.

### Project structure

```
.
├── src/
│   ├── train/whisper_lora_finetune/
│   │   ├── train.py                 # training
│   │   ├── diagnose_runs.py         # compare trainings
│   │   ├── segment_long_audio.py    # inspect how long recordings are cut
│   │   ├── requirements.txt
│   │   └── whisper_finetune/        # training code
│   └── evaluation/evaluation/
│       ├── evaluate.py              # test BASE vs your model
│       ├── compare_results.py       # compare two test results
│       └── whisper_eval/            # test code and settings
├── data/
│   ├── train/   (data.csv + audio/)
│   └── test/    (test.csv + audio/<person>/)
├── run_lora.sh                      # train many settings at once
└── LICENSE
```

---

## About the author

**[Paulo Condado](https://paulocondado.net)** is an Assistant Researcher at
[CENSE, NOVA FCT](https://www.cense.fct.unl.pt/) (Portugal), working on assistive technologies,
human–computer interaction and machine learning. He has cerebral palsy, and advocates a
constructionist approach in which people with disabilities take an active part in building their
own assistive technologies.

## Research

- A paper describing this work is available on **arXiv** and has been submitted to a prestigious peer-reviewed journal in the field.

>Condado P. A & Lobo F. G. (2026). DIY Customized Speech Recognition for a Person with Speech Disabilities (arXiv Report No. 2611.11327). arXiv. https://arxiv.org/abs/2611.11327

If you use this project in your work, please cite the papers above.

## License

**MIT License** — Copyright © 2026 Paulo Condado. You can use, change and share this code freely;
see [LICENSE](LICENSE).

The tools this project uses are installed separately and keep their own licenses: PyTorch,
NumPy, pandas, SciPy, soundfile, torchcodec (BSD), Transformers, Datasets, PEFT, Accelerate,
jiwer (Apache-2.0), librosa (ISC), epitran (MIT), edge-tts (LGPL-3.0), soxr and libsndfile
(LGPL-2.1). The Whisper models are © OpenAI and downloaded under their own license: if you share
a trained model, credit OpenAI Whisper and keep its license notice. Recordings and transcriptions
are not covered by this license.
