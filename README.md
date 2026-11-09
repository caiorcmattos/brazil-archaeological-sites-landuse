# Cultural Heritage at Risk: Quantifying the threats of land use change to Brazil's Archaeological Sites

[![Paper DOI](https://img.shields.io/badge/paper-<PAPER--DOI>-blue)](https://doi.org/<PAPER-DOI>)
[![Data DOI](https://img.shields.io/badge/data-10.5281%2Fzenodo.24086255-blue)](https://doi.org/10.5281/zenodo.24086255)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code and analysis pipeline for the paper:
> **Mattos, C., Hirota, M., Prestes Carneiro, G., Schmidt, M. J., Trindade, T. B., Neves, E. G., da Silva, C. A., Fuentes, A., Mugge, M., Simon de Moraes, R., Rosa, M. R., Shimbo, J., Biehl, J.** (<YEAR>). *Cultural Heritage at Risk: Quantifying the threats of land use change to Brazil's Archaeological Sites*. Frontiers in Conservation Science. https://doi.org/<PAPER-DOI>

## 📌 Project Overview
We measured how the landscapes around Brazil's recorded archaeological sites changed between 1985 and 2025. The national registry of the Brazilian heritage institute (IPHAN) was filtered to 32,285 continental sites, and land cover within 100 m of each registered point was measured with MapBiomas Brazil Collection 11 and compared with the artificial water bodies mapped by the national water agency (ANA). Natural cover around the sites fell from 57% to 43.6% and anthropic cover rose from 37.5% to 47.1%, most of it before 2004. 2,477 sites overlap reservoirs, and 988 of them have buffers entirely classified as water, almost all at hydropower reservoirs. Buffers of 200, 500 and 1,000 m test the sensitivity of these results.

## 📂 Repository Structure
```text
├── config/             # Parameters (TOML), manual filtering decisions, MapBiomas C11 legend (Table S1)
├── data/               # Not versioned; data/README.md lists the inputs and their sources
├── scripts/            # Pipeline, 01–14, run in order; make_figures.py draws the paper figures
├── src/sitios_arq/     # Package: filtering, buffers, extraction, aggregation, hexagons, reservoirs, figures
├── tests/              # Unit tests on synthetic data
├── assets/fonts/       # IBM Plex Sans (SIL Open Font License), used in the figures
├── environment.yml     # Conda environment
└── README.md
```

## ⚙️ Installation & Setup
```bash
git clone https://github.com/caiorcmattos/brazil-archaeological-sites-landuse.git
cd sitios-arqueologicos
conda env create -f environment.yml
conda activate sitios-arq
pip install -e .
python -m pytest
```

## 🚀 How to Reproduce Findings
1. **Download Data.** The data package on Zenodo ([10.5281/zenodo.24086255](https://doi.org/10.5281/zenodo.24086255)) holds the IPHAN snapshot, the filtered sites, the land-cover extraction per site and every derived table. Point the code to it:
   ```bash
   cd /path/to/zenodo-package
   unzip raw.zip && unzip processed.zip    # recreates raw/ and processed/
   export SITIOS_DATA_ROOT=$PWD
   ```
   To run the full pipeline from the original sources instead, download the inputs listed in [`data/README.md`](data/README.md) (MapBiomas Collection 11 rasters, IBGE and ANA layers).

2. **Run Pipeline.** With the Zenodo package, the summary tables are rebuilt with:
   ```bash
   python scripts/14_summary_tables.py
   ```
   From the original sources, run `scripts/01_prepare_ibge_layers.py` to `scripts/14_summary_tables.py` in order (each takes `--help`). The land-cover extraction (`06_extract_mapbiomas.py`, `11_extract_hex.py`) is split into independent tasks, one per raster, that can run in parallel.

3. **Generate Paper Figures.** Figures 1–5, S1 and S2 are drawn from the summary tables into `outputs/figures/` (PDF, TIFF and PNG):
   ```bash
   python scripts/make_figures.py --all       # or --fig 2, --fig S1
   ```

## 🙏 Acknowledgments
This work was funded by the "Indigenizing Conservation Science" project, sponsored by the Princeton "New Ideas in the Social Sciences" seed grant from the Office of the Dean for Research and supported by the High Meadows Environmental Institute's Thomas A. & Currie C. Barron Family Biodiversity Research Challenge Fund. M.H. was supported by the Serrapilheira Institute (grant Serra-1709-18983). We thank the Instituto do Patrimônio Histórico e Artístico Nacional (IPHAN) for curating the national database of archaeological sites, and the MapBiomas initiative for the land-cover data and for help in conceptualizing this study.

## ✉️ Contact & Support
For questions regarding the data or methodology, please contact:
* **Primary Contact:** Caio Mattos (caio.mattos@relva.org)

Code is released under the MIT license ([`LICENSE`](LICENSE)); the data package under CC BY 4.0.
