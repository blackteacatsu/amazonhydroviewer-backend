# Amazon Hydroviewer Backend

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Backend data processing and storage infrastructure for the [Amazon Hydroclimate Viewer](https://hydroclimate.shinyapps.io/amazonhydroviewer/) application. This repository manages the collection, processing, and staging of hydrological forecast data for the Amazon basin.

## Overview

The Amazon Hydroviewer Backend processes NASA Land Data Assimilation System (LDAS) forecast data to provide near real-time hydroclimate information for regions within the Amazon basin. The system computes zonal statistics across hydrological basins (HydroBASINS level 5) and generates time-series data for visualization in the companion web application.

### Key Features

- **Automated Data Processing**: Processes multiple LDAS variables including evapotranspiration, humidity, runoff, precipitation, soil moisture, and soil temperature
- **Zonal Statistics**: Computes basin-level statistics (mean/max) for each HydroBASINS polygon
- **Probabilistic Forecasts**: Generates ensemble-based probabilistic forecasts from LDAS data
- **Climatology Analysis**: Calculates baseline climatology for anomaly detection and comparison
- **CSV Export**: Outputs processed data in time-series CSV format for efficient downstream consumption

### Processed Variables

- **Evap_tavg**: Evapotranspiration (time average)
- **Qair_f_tavg**: Specific humidity (time average)
- **Qs_tavg**: Surface runoff (time average)
- **Rainf_tavg**: Rainfall rate (time average)
- **SoilMoist_inst**: Soil moisture (instantaneous, 4 vertical levels)
- **SoilTemp_inst**: Soil temperature (instantaneous, 4 vertical levels)

## Repository Structure

```
amazon_hydroviewer_backend/
├── core/                                    # Core processing notebooks
│   ├── get_ldas_base_climatology.ipynb     # Climatology baseline computation
│   └── get_ldas_probabilistics_data.ipynb  # Probabilistic forecast generation
├── get_ldas_probabilistics_output/          # Probabilistic forecast output
├── get_zonal_averages_csv/                  # Processed zonal average time-series (tabulated)
├── get_zonal_averages_climatology_csv/      # Climatology baseline data (tabulated)
├── get_zonal_average.py                     # Zonal statistics computation script
├── get_ldas_base_climatology.ipynb          # Climatology analysis notebook
├── get_ldas_probabilistics_data.ipynb       # Probabilistic forecast notebook
├── parting_raw_forecast.ipynb               # Raw forecast data processing
├── scrutinize.ipynb                         # Data quality analysis notebook
└── requirements.txt                         # Python dependencies
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/amazon_hydroviewer_backend.git
   cd amazon_hydroviewer_backend
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Computing Zonal Averages [This current version removes this]

The [get_zonal_average.py](get_zonal_average.py) script processes NetCDF forecast files and computes zonal statistics for each HydroBASINS polygon:

```bash
python get_zonal_average.py
```

**Output**: CSV files in `get_zonal_averages_csv/` directory, one per PFAF basin ID containing time-series data for all processed variables.

### Processing Probabilistic Forecasts

Run the probabilistic forecast notebook to generate ensemble-based forecasts:

```bash
jupyter notebook get_ldas_probabilistics_data.ipynb
```

### Generating Climatology Baseline

To compute baseline climatology for anomaly analysis:

```bash
jupyter notebook get_ldas_base_climatology.ipynb
```

## Data Sources

- **LDAS Forecast Data**: NASA Land Data Assimilation System forecasts
- **HydroBASINS**: [Level 5 basin boundaries for South America](https://github.com/blackteacatsu/spring_2024_envs_research_amazon_ldas/raw/main/resources/hybas_sa_lev05_areaofstudy.geojson)

## Dependencies

Core Python libraries:
- `xarray`: Multi-dimensional array processing
- `geopandas`: Geospatial data manipulation
- `regionmask`: Spatial mask generation
- `pandas`: Data analysis and CSV I/O
- `tqdm`: Progress bar utilities

See [requirements.txt](requirements.txt) for complete dependency list.

## Output Format

Processed data is exported as CSV files with the following structure:

| time | Variable1 | Variable2 | ... | PFAF_ID |
|------|-----------|-----------|-----|---------|
| 2024-12-01T00:00:00 | 2.34 | 15.67 | ... | 6510000010 |
| 2024-12-01T03:00:00 | 2.45 | 15.89 | ... | 6510000010 |

Each row represents a time step, with columns for each processed variable and a PFAF_ID identifier.

## Related Projects

- **Amazon Hydroclimate Viewer** (Frontend): [https://hydroclimate.shinyapps.io/amazonhydroviewer/](https://hydroclimate.shinyapps.io/amazonhydroviewer/)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support, please open an issue in the GitHub repository.

## Acknowledgments

- NASA Land Data Assimilation System for forecast data
- HydroBASINS for basin boundary data
- The open-source geospatial Python community
