# MorphoMapping Launcher — Windows & macOS

One-click install and start scripts for the MorphoMapping GUI (Imaging Flow Cytometry analysis). Run the install script once, then use the start script whenever you want to open the app.

---

## Get the scripts

This folder lives in the [MorphoMapping repository](https://github.com/Wguido/MorphoMapping). To use it:

**Option A — Download**

1. Open the [launcher folder on GitHub](https://github.com/Wguido/MorphoMapping/tree/main/launcher).
2. Click **Code** → **Download ZIP** to download the whole repo, then go to the `launcher` folder inside the ZIP.
3. Use the steps below for your OS.

**Option B — Clone the repo**

```bash
git clone https://github.com/Wguido/MorphoMapping.git
cd MorphoMapping/launcher
```

Then run the install/start scripts for your OS as described below.

---

## Scripts

| Purpose | Windows | macOS |
|--------|---------|--------|
| **Install** (run once) | `Install_MorphoMapping.bat` | `Install_MorphoMapping.sh` |
| **Start GUI** | `Start_MorphoMapping.bat` | `Start_MorphoMapping.sh` |

---

## Windows (10 / 11)

**Prerequisite:** Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if needed (PowerShell as Administrator):

```powershell
winget install -e --id Anaconda.Miniconda3 --accept-package-agreements --accept-source-agreements
```

Then open a **new** terminal window.

**Steps:**

1. Double-click **Install_MorphoMapping.bat** and wait until it says “Installation complete”.
2. Double-click **Start_MorphoMapping.bat** to launch the GUI.

**Without Git:** Download [MorphoMapping as ZIP](https://github.com/Wguido/MorphoMapping/archive/refs/heads/main.zip), extract it to `%USERPROFILE%\MorphoMapping` (so that the `gui` folder is directly inside), then run **Install_MorphoMapping.bat** again.

---

## macOS (OS X)

**Prerequisite:** Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) if needed (`brew install --cask miniconda`), or answer “y” when the install script offers to download it.

**Steps:**

```bash
cd /path/to/MorphoMapping/launcher
chmod +x Install_MorphoMapping.sh Start_MorphoMapping.sh
./Install_MorphoMapping.sh
./Start_MorphoMapping.sh
```

You can also double-click **Start_MorphoMapping.sh** in Finder after making it executable (e.g. via Terminal as above).

**Without Git:** Download [MorphoMapping as ZIP](https://github.com/Wguido/MorphoMapping/archive/refs/heads/main.zip), extract and copy the contents to `~/MorphoMapping` (so that `gui/morphomapping_gui.py` exists there), then run **Install_MorphoMapping.sh** again.

---

## Change install location

- **Windows:** Edit `INSTALL_DIR` in both `.bat` files (default: `%USERPROFILE%\MorphoMapping`).
- **macOS:** Edit `INSTALL_DIR` in both `.sh` files (default: `~/MorphoMapping`).

---

## Troubleshooting

| Issue | Windows | macOS |
|-------|---------|--------|
| Conda not found | Install Miniconda via winget (see above), open a new terminal, run the install script again. | Run `brew install --cask miniconda` or answer “y” when the script asks to install Miniconda. |
| MorphoMapping not found | Set `INSTALL_DIR` in both `.bat` files to the folder that contains `gui\morphomapping_gui.py`. | Set `INSTALL_DIR` in both `.sh` files to the folder that contains `gui/morphomapping_gui.py`. |
| GUI does not start | Leave the script window open to read the error; try running **Install_MorphoMapping.bat** again. | Run `./Start_MorphoMapping.sh` in Terminal and read the error message. |
| “Permission denied” (macOS) | — | Run `chmod +x Install_MorphoMapping.sh Start_MorphoMapping.sh`. |

---

## Sharing

Share the **launcher** folder (or the whole MorphoMapping repo). Recipients use the Windows or macOS scripts as described above.
