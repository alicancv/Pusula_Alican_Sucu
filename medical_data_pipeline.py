import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from pandas.api.types import is_numeric_dtype

class MedicalDataPipeline:
    def __init__(self):
        self.data = None
        self.processed_data = None
    
    # Loading data
    def load_data(self, file_path):
        try:
            if file_path.endswith(".xlsx"):
                self.data = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format")
                
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    # Displaying basic information about data    
    def basic_info(self, df):
        print("-"*50)
        print("DATASET OVERVIEW")
        print("-"*50)

        report = {
            "Dataset Shape": df.shape if df is not None else "df is None",
            "Dataset Columns": list(df.columns),
            "Dataset Duplicate Rows": df.duplicated().sum(),
            "Dataset Missing Values": df.isnull().sum().sum(),
            "Dataset Data Types": dict(df.dtypes),
            "Non-Null Count": df.count(),
            "Null Count": df.isnull().sum(),
            "Null %": (df.isnull().sum() / len(df) * 100).round(2),
            "Dataset Memory Usage ": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        }

        for key, value in report.items():
            print(f"{key}: {value}\n")
    
    # Target Attribute Analysis
    def analyze_target_attribute(self, df, target_col = "TedaviSuresi"):
        print("-"*50)
        print("TARGET ATTRIBUTE ANALYSIS")
        print("-"*50)

        if target_col not in df.columns:
            print(f"Column {target_col} not found")
            return None            

        print("-" * 30)
        print("Statistical Summary:")
        print(df[target_col].describe())
        print("-" * 30)
        
        print("-" * 30)
        print(f"Value Distribution:")
        print(df[target_col].value_counts().sort_index())
        print("-" * 30)

        if is_numeric_dtype(df[target_col].dtype):
            # Visualization
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))

            # Histogram
            df[target_col].hist(bins=20, ax=axes[0], edgecolor="black", alpha=0.7)
            axes[0].axvline(df[target_col].mean(), color="red", linestyle="--", 
                           label=f"Mean: {df[target_col].mean():.2f}")
            axes[0].axvline(df[target_col].median(), color="green", linestyle="--", 
                           label=f"Median: {df[target_col].median():.2f}")
            axes[0].set_title("Distribution of Treatment Duration")
            axes[0].set_xlabel("Treatment Duration (Sessions)")
            axes[0].set_ylabel("Frequency")
            axes[0].legend()

            # Box plot
            df[target_col].plot(kind="box", ax=axes[1])
            axes[1].set_title("Box Plot of Treatment Duration")
            axes[1].set_ylabel("Treatment Duration (Sessions)")

            plt.tight_layout()
            plt.show()
    
    # Missing Data Analysis
    def analyze_missing_data(self, df):
        print("-"*50)
        print("\nMISSING DATA ANALYSIS")
        print("-"*50)

        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100

        missing_df = pd.DataFrame({
            "Missing Count": missing_data,
            "Missing Percentage": missing_percent
        }).sort_values("Missing Percentage", ascending=False)

        if len(missing_df) > 0:
            print(missing_df)
            
            # Visualization
            plt.figure(figsize=(12, 6))

            # Missing data heatmap
            plt.subplot(1, 2, 1)
            sns.heatmap(df.isnull(), cmap="viridis", cbar=True, yticklabels=False)
            plt.title("Missing Data Pattern")

            # Missing data bar plot
            plt.subplot(1, 2, 2)
            missing_df["Missing Percentage"].plot(kind="bar")
            plt.title("Missing Data Percentage by Column")
            plt.xlabel("Columns")
            plt.ylabel("Missing Percentage (%)")
            plt.xticks(rotation=45)

            plt.tight_layout()
            plt.show()

            print(f"Total Missing Values: {missing_data.sum()}")
            print(f"Complete Rows: {len(df.dropna())} ({len(df.dropna())/len(df)*100:.2f}%)")
        else:
            print("No missing data found!")

        return missing_df

    # Categorical Attribute Analysis
    def analyze_categorical_attributes(self, df):
        print("-"*50)
        print("CATEGORICAL ATTRIBUTE ANALYSIS")
        print("-"*50)

        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
        
        for col in categorical_cols:
            if col in df.columns:
                print(f"\n{col.upper()}:")
                print("-" * (len(col) + 10))
                
                value_counts = df[col].value_counts(dropna=False)
                percentages = df[col].value_counts(normalize=True, dropna=False) * 100

                analysis_df = pd.DataFrame({
                    "Count": value_counts,
                    "Percentage": percentages.round(2)
                })
                print(analysis_df)
                print("-" * (len(col) + 10))
        
        # Visualizations
        n_categorical = len(categorical_cols)
        if n_categorical > 0:
            n_rows = (n_categorical + 1) // 2
            fig, axes = plt.subplots(n_rows, 2, figsize=(15, 5 * n_rows))

            if n_categorical == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            for i, col in enumerate(categorical_cols):
                if i < len(axes):
                    value_counts = df[col].value_counts()

                    if len(value_counts) <= 5:
                        # Pie chart for few categories
                        axes[i].pie(value_counts.values, labels=value_counts.index, 
                                  autopct="%1.1f%%", startangle=90)
                        axes[i].set_title(f"Distribution of {col}")
                    else:
                        # Bar plot for many categories
                        value_counts.head(10).plot(kind="bar", ax=axes[i])
                        axes[i].set_title(f"Distribution of {col}")
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel("Count")
                        axes[i].tick_params(axis="x", rotation=45)

            # Hide unused subplots
            for i in range(len(categorical_cols), len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            plt.show()
                
    # Numerical Attribute Analysis  
    def analyze_numerical_attributes(self, df):
        print("-"*50)
        print("NUMERICAL ATTRIBUTE ANALYSIS")
        print("-"*50)

        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Outliers
        for col in numerical_cols:
            print(f"\n{col} Statistics:")
            print(df[col].describe())

            outliers = self.outlier_analysis(df[col])
            print(f"{col} outliers (IQR method): {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")

        # Correlation
        self.correlation_analysis(df)

        # Visualization
        if len(numerical_cols) > 0:
            n_rows = (len(numerical_cols) + 2) // 3
            fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows))
            axes = axes.flatten() if len(numerical_cols) > 1 else [axes]

            for i, col in enumerate(numerical_cols):
                if i < len(axes):
                    df[col].hist(bins=20, ax=axes[i], alpha=0.7, edgecolor="black")
                    axes[i].axvline(df[col].mean(), color="red", linestyle="--", 
                                  label=f"Mean: {df[col].mean():.2f}")
                    axes[i].axvline(df[col].median(), color="green", linestyle="--", 
                                  label=f"Median: {df[col].median():.2f}")
                    axes[i].set_title(f"Distribution of {col}")
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel("Frequency")
                    axes[i].legend()

            # Hide unused subplots
            for i in range(len(numerical_cols), len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            plt.show()

    def outlier_analysis(self, col):
        if is_numeric_dtype(col):
            # Outlier detection using IQR
            Q1 = col.quantile(0.25)
            Q3 = col.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_mask = (col < lower_bound) | (col > upper_bound)
            outliers = col[outlier_mask]
            return outliers

        return None 

    def correlation_analysis(self, df):
        print("-"*30)
        print("CORRELATION ANALYSIS")
        print("-"*30)

        numerical_df = df.select_dtypes(include=[np.number])

        if len(numerical_df.columns) < 2:
            print("Not enough numeric columns for correlation analysis")
            return

        # Calculate correlation matrix
        corr_matrix = numerical_df.corr()

        print("Correlation Matrix:")
        print(corr_matrix.round(3))

        # Visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0,
                    square=True, fmt=".3f", cbar_kws={"label": "Correlation"})
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()
        plt.show()

        return corr_matrix

    # Complex Attribute Analysis
    def complex_attribute_analysis(self, df, col):
        print("="*30)
        print(f"COMPLEX ATTRIBUTE ANALYSIS ({col})")
        print("="*30)

        if col not in df.columns:
            print(f"Column {col} not found")
            return

        col_without_nan = df[col].dropna()
        print(f"Occupancy Rate: {len(col_without_nan)} ({len(col_without_nan)/len(df)*100:.2f}%)")

        # Extract individual values
        all_values = []
        for strings in col_without_nan:
            if isinstance(strings, str):
                all_values.extend([s.strip() for s in strings.split(",")])

        if all_values:
            attr_value_counts = pd.Series(all_values).value_counts().head(10)
            print(f"\nTop 10 Most Common Values of {col}")
            print(attr_value_counts)

            # Visualization
            plt.figure(figsize=(12, 6))
            attr_value_counts.plot(kind="bar")
            plt.title(f"\nTop 10 Most Common Values of {col}")
            plt.xlabel("Value")
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()

        return all_values

    #EDA
    def exploratory_data_analysis(self, df, target_attr):
        print("-"*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("-"*50)

        # Target Attribute Analysis
        self.analyze_target_attribute(df, target_attr)

        # Missing Data Analysis
        self.analyze_missing_data(df)

        # Categorical Attribute Analysis
        self.analyze_categorical_attributes(df)
        
        # Numerical Attribute Analysis  
        self.analyze_numerical_attributes(df)

        # Complex Attribute Analysis - KronikHastalik Analysis
        self.complex_attribute_analysis(df, "KronikHastalik")

        # Complex Attribute Analysis - Alerji analysis
        self.complex_attribute_analysis(df, "Alerji")

        # Complex Attribute Analysis - Bolum analysis
        self.complex_attribute_analysis(df, "Bolum")
    
    def extract_numeric_attributes(self, df, cols):
        for col in cols:
            if df is not None and col in df.columns and df[col].dtype == "object":
                df[f"{col}_numeric"] = df[col].str.extract(r"(\d+)").astype(float)
                df.drop(col, axis=1, inplace=True)
                print(f"{col} converted to numeric")

    def encode_categorical_attributes(self, df, cols):
        label_encoders = {}

        for col in cols:
            if df is not None and col in df.columns:
                le = LabelEncoder()
                df[f"{col}_encoded"] = le.fit_transform(df[col].fillna("Unknown"))
                label_encoders[col] = le
                print(f"{col} encoded")

    def process_complex_categorical_attributes(self, df, cols, min_frequency = 0.15):
        for col in cols:
            # Getting possible values of col
            all_values = []
            not_nan_data = df[col].dropna()

            for values in not_nan_data:
                if isinstance(values, str):
                    temp = [v.strip() for v in values.split(',')]
                    all_values.extend(temp)

            # Finding more common values of col
            value_counts = pd.Series(all_values).value_counts()
            min_count = len(df) * min_frequency
            common_values = value_counts[value_counts >= min_count].index.tolist()

            # Creating binary attributes for common values of col
            for value in common_values:
                clean_name = value.replace(' ', '_')
                df[f'has_{clean_name}'] = df[col].fillna('').str.contains(value, case=False).astype(int)

            # Creating an extra total col feature
            df[f"total_{col}"] = df[col].fillna('').apply(
                lambda x: len([d.strip() for d in x.split(',') if d.strip()]) if x else 0
            )

    def preprocess_data(self):
        print("-"*50)
        print("DATA PREPROCESSING")
        print("-"*50)

        self.processed_data = self.data.copy()
        print(f"Original shape: {self.processed_data.shape}")

        print("\n1. Attributes that should be numeric are converted to numeric")
        self.extract_numeric_attributes(self.processed_data, ["TedaviSuresi", "UygulamaSuresi"])

        print("\n2. Encoding categorical attributes")
        self.encode_categorical_attributes(self.processed_data, ["Cinsiyet", "KanGrubu", "Uyruk"])

        print("\n3. Processing complex attributes")
        self.process_complex_categorical_attributes(self.processed_data, ["KronikHastalik", "Alerji", "Bolum"])

        print("\n4. Creating Yas_Group Attribute")
        if "Yas" in self.processed_data.columns:
            self.processed_data["Yas_Group"] = pd.cut(self.processed_data["Yas"], 
                                             bins=[0, 18, 30, 50, 70, 100], 
                                             labels=[0, 1, 2, 3, 4],
                                             include_lowest=True)

        print("\n5. Handling missing values")
        numeric_cols = self.processed_data.select_dtypes(include=[np.number]).columns

        imputer = SimpleImputer(strategy="median")
        self.processed_data[numeric_cols] = imputer.fit_transform(self.processed_data[numeric_cols])

        print("\n6. Attribute scaling")
        scaler = StandardScaler()
        scale_cols = ["Yas", "UygulamaSuresi_numeric"]
        scale_cols = [col for col in scale_cols if col in self.processed_data.columns]

        if scale_cols:
            self.processed_data[scale_cols] = scaler.fit_transform(self.processed_data[scale_cols])

        # Tanilar, TedaviAdi and UygulamaYerleri attributes are removed because they are highly correlated to TedaviSuresi and they prevent model to
        # learn how to predict TedaviSuresi with other features.
        print("\n7. Removing unnecessary columns")
        cols_to_drop = ["HastaNo", "KronikHastalik", "Alerji", "Tanilar", "TedaviAdi", "UygulamaYerleri", "Bolum"]
        cols_to_drop.extend(["Cinsiyet", "KanGrubu", "Uyruk"])

        existing_cols_to_drop = [col for col in cols_to_drop if col in self.processed_data.columns]
        self.processed_data.drop(existing_cols_to_drop, axis=1, inplace=True, errors="ignore")

        print("\n8. Removing duplicate rows")
        if self.processed_data.duplicated().sum() > 0:
            self.processed_data.drop_duplicates(inplace = True)

        return self.processed_data
    
    def data_quality_report(self):
        print("-"*50)
        print("DATA QUALITY REPORT")
        print("-"*50)

        print("-"*50)
        print("ORIGINAL DATA")
        self.basic_info(self.data)
        print("-"*50)
        print("PROCESSED DATA")
        self.basic_info(self.processed_data)
        print("-"*50)

        self.exploratory_data_analysis(self.processed_data, "TedaviSuresi_numeric")

    def run_complete_pipeline(self, file_path):
        print("STARTING PUSULA MEDICAL DATA ANALYSIS PIPELINE")
        print("-"*50)
        
        self.load_data(file_path)
        if self.data is None:
            return
        
        self.basic_info(self.data)
        
        self.exploratory_data_analysis(self.data, "TedaviSuresi")
        
        self.preprocess_data()
        
        self.data_quality_report()
        
        print("\nPIPELINE COMPLETED SUCCESSFULLY!")
        print("-"*60)
        
        return self.processed_data

if __name__ == "__main__":
    pipeline = MedicalDataPipeline()

    processed_data = pipeline.run_complete_pipeline("Talent_Academy_Case_DT_2025.xlsx")