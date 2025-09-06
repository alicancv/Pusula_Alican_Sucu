# Pusula Medical Data Analysis Pipeline

**Name:** Alican 
**Surname:** Sucu  
**Email:** alicansucu@outlook.com

## Project Overview

This project implements a comprehensive data analysis pipeline for a physical medicine & rehabilitation dataset as part of the Pusula Data Science Intern Case Study 2025. The pipeline performs in-depth Exploratory Data Analysis (EDA) and data preprocessing to prepare the dataset for potential predictive modeling with `TedaviSuresi` (Treatment Duration) as the target variable.

The dataset contains 2,235 observations and 13 features related to patient demographics, medical conditions, and treatment information.

## Dataset Description

| Column | Description |
|--------|-------------|
| HastaNo | Anonymized patient ID |
| Yas | Age |
| Cinsiyet | Gender |
| KanGrubu | Blood type |
| Uyruk | Nationality |
| KronikHastalik | Chronic conditions (comma-separated list) |
| Bolum | Department/Clinic |
| Alerji | Allergies (single or comma-separated) |
| Tanilar | Diagnoses |
| TedaviAdi | Treatment name |
| **TedaviSuresi** | **Treatment duration in sessions (TARGET)** |
| UygulamaYerleri | Application sites |
| UygulamaSuresi | Application duration |

## Features

- **Comprehensive EDA**: Statistical analysis, visualization, and pattern detection
- **Missing Data Analysis**: Identification and visualization of missing data patterns
- **Categorical Analysis**: Distribution analysis with appropriate visualizations
- **Numerical Analysis**: Statistical summaries, outlier detection, and correlation analysis
- **Complex Attribute Processing**: Handling of multi-value categorical fields
- **Feature Engineering**: Creation of new meaningful features
- **Data Preprocessing**: Complete data cleaning and preparation pipeline
- **Modular Design**: Reusable pipeline architecture

## Installation

### Prerequisites
- Python 3.7+
- Required libraries (install via pip):

## Usage

### Basic Usage
```python
from medical_pipeline import MedicalDataPipeline

# Initialize the pipeline
pipeline = MedicalDataPipeline()

# Run complete analysis
processed_data = pipeline.run_complete_pipeline("Talent_Academy_Case_DT_2025.xlsx")
```

### Step-by-Step Usage
```python
# Load data
pipeline.load_data("Talent_Academy_Case_DT_2025.xlsx")

# Perform EDA
pipeline.exploratory_data_analysis(pipeline.data, "TedaviSuresi")

# Preprocess data
processed_data = pipeline.preprocess_data()

# Generate quality report
pipeline.data_quality_report()
```

## Pipeline Architecture

```
MedicalDataPipeline
├── Data Loading & Validation
├── Basic Information Analysis
├── Exploratory Data Analysis
│   ├── Target Attribute Analysis
│   ├── Missing Data Analysis
│   ├── Categorical Attribute Analysis
│   ├── Numerical Attribute Analysis
│   └── Complex Attribute Analysis
├── Data Preprocessing
│   ├── Numeric Conversion
│   ├── Categorical Encoding
│   ├── Complex Attribute Processing
│   ├── Feature Engineering
│   ├── Missing Value Imputation
│   ├── Feature Scaling
│   └── Data Cleaning
└── Data Quality Reporting
```

## Key Methods

### Data Loading
- `load_data(file_path)`: Loads Excel files with error handling

### EDA Methods
- `basic_info(df)`: Comprehensive dataset overview
- `analyze_target_attribute(df, target_col)`: Target variable analysis with visualizations
- `analyze_missing_data(df)`: Missing data patterns and visualization
- `analyze_categorical_attributes(df)`: Categorical variable distribution analysis
- `analyze_numerical_attributes(df)`: Numerical analysis with outlier detection
- `complex_attribute_analysis(df, col)`: Multi-value categorical field analysis
- `correlation_analysis(df)`: Correlation matrix and heatmap

### Preprocessing Methods
- `extract_numeric_attributes(df, cols)`: Convert string numbers to numeric
- `encode_categorical_attributes(df, cols)`: Label encoding for categorical variables
- `process_complex_categorical_attributes(df, cols)`: Binary feature creation from multi-value fields
- `preprocess_data()`: Complete preprocessing pipeline

### Utility Methods
- `outlier_analysis(col)`: IQR-based outlier detection
- `data_quality_report()`: Before/after preprocessing comparison
- `run_complete_pipeline(file_path)`: End-to-end pipeline execution

## Preprocessing Steps

1. **Numeric Conversion**: Extract numeric values from string fields
2. **Categorical Encoding**: Label encode categorical variables
3. **Complex Attribute Processing**: Create binary features from multi-value fields
4. **Feature Engineering**: Age grouping and derived features
5. **Missing Value Imputation**: Median imputation for numeric variables
6. **Feature Scaling**: StandardScaler for selected numeric features
7. **Data Cleaning**: Remove unnecessary columns and duplicates

## Output

The pipeline generates:
- Comprehensive EDA visualizations
- Statistical summaries and insights
- Processed dataset ready for modeling
- Data quality comparison reports

## File Structure

```
Pusula_Name_Surname/
├── README.md
├── medical_data_pipeline.py
├── Medical_Data_Analysis_Documentation.md
├── Talent_Academy_Case_DT_2025.xlsx
└── requirements.txt
```

## Key Findings

- **Target Variable**: Treatment duration shows specific distribution patterns
- **Missing Data**: Systematic missing data patterns identified and addressed
- **Feature Engineering**: Created meaningful binary indicators from complex categorical fields
- **Data Quality**: Improved data consistency and removed duplicates
- **Feature Scaling**: Applied appropriate scaling for machine learning readiness

## Technical Specifications

- **Language**: Python 3.7+
- **Key Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn
- **Input Format**: Excel (.xlsx)
- **Output**: Processed pandas DataFrame
- **Memory Efficient**: Optimized for large datasets

## Future Enhancements

- Advanced outlier treatment methods
- Automated feature selection
- Interactive visualizations with Plotly
- Model integration capabilities
- Configuration file support

## Case Study Compliance

This project fully addresses the Pusula Data Science Intern Case Study 2025 requirements:

✅ **Exploratory Data Analysis**: Comprehensive EDA with Python, Pandas, Matplotlib, and Seaborn  
✅ **Data Preprocessing**: Complete preprocessing pipeline for model readiness  
✅ **Documentation**: Detailed documentation of findings and methodology  
✅ **Pipeline Architecture**: Modular, reusable code structure  
✅ **GitHub Repository**: Proper repository structure with README.md  

## Contact Information

For questions or clarifications regarding this project, please contact:
- **Email**: alicansucu@outlook.com
- **GitHub**: https://github.com/alicancv/

---

**Submission Date**: September 6, 2025  
**Project**: Pusula Data Science Intern Case Study 2025