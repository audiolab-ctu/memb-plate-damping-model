# README

This repository contains the Python implementation of the analytical method proposed in the article:

**Analytical Modeling of MEMS Transducers with Circular Membranes and Plates Damped by an Air Gap with Central Opening.**

The repository includes three Python scripts and several text files. The text files provide reference numerical results exported from COMSOL Multiphysics and are stored in the `Comsol_data` directory. The Python scripts are:

* **functionsMembPlate.py**
  Contains the functions required to perform the calculations. Each function is documented with comments. In some cases, uncommenting specific lines provides alternative results (e.g., lines 418 and 425 allow the calculation of the mean displacement over the rigid wall area only, excluding the central hole).

* **DisplacementPressure_Space.py**
  Generates figures corresponding to Figs. 2–5 in the article, showing the real and imaginary parts of the displacement of the moving component and the acoustic pressure behind it as a function of the radial coordinate.

  * After setting the transducer and air parameters, COMSOL result filenames are specified. These are selected by commenting/uncommenting lines. The files are expected to be located in the `Comsol_data` directory.
  * Display of COMSOL reference results is controlled by the variable `display_Comsol` (line 41). If set to `False`, only analytical results are shown.
  * The membrane or plate approach is selected on line 38.
  * The frequencies for the calculation are then defined inside the if/else part.
  * Line 44 provides an option to save the figures (both `.eps` and `.png` formats).

* **MeanDisplacement_Frequency.py**
  Generates figures corresponding to Figs. 6–9 in the article, showing the mean displacement as a function of frequency.

  * After setting the transducer and air parameters, the membrane or plate approach is selected on line 33.
  * The frequency points are then defined in the script.
  * COMSOL result filenames are specified similarly as above and are expected in the `Comsol_data` directory.
  * Display of COMSOL reference results is controlled by the variable `display_Comsol` (line 49). If set to `False`, only analytical results are shown.

---

## Requirements

The code was tested with:

* Python 3.12

Required Python packages:

* numpy
* matplotlib
* scipy
* alive-progress

Install dependencies using:

```bash
pip install numpy matplotlib scipy alive-progress
```

or using the provided requirements file:

```bash
pip install -r requirements.txt
```

The following modules are part of the Python standard library and require no installation:

* concurrent.futures
* math
* pathlib


---

## Usage

To reproduce the figures presented in the article:

* Run `DisplacementPressure_Space.py` for spatial distributions (Figs. 2–5)
* Run `MeanDisplacement_Frequency.py` for frequency responses (Figs. 6–9)

The inclusion of COMSOL reference results can be enabled or disabled using the `display_Comsol` variable in each script.

All model parameters can be modified directly in the scripts.

---

## Repository structure

* `functionsMembPlate.py` – core computational functions
* `DisplacementPressure_Space.py` – spatial distributions (Figs. 2–5)
* `MeanDisplacement_Frequency.py` – frequency response (Figs. 6–9)
* `Comsol_data/` – reference COMSOL data used for comparison

---

## Notes

Reference data from COMSOL Multiphysics are included as text files in the `Comsol_data` directory and are used for comparison with the analytical model.

---

## Citation

If you use this code, please cite:

Honzik et al.,
*Analytical Modeling of MEMS Transducers with Circular Membranes and Plates Damped by an Air Gap with Central Opening*,
Journal of Sound and Vibration, 2026.

---

## License

This project is licensed under the MIT License.

---

**Please do not hesitate to contact the first author at [honzikp@fel.cvut.cz](mailto:honzikp@fel.cvut.cz) if you have any questions.**
