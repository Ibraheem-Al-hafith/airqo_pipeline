import streamlit as st
import pandas as pd
import numpy as np
from src.config import Config
from src.inference.predict import make_predictions

# ==========================================
# 1. SAMPLE DATA SETUP
# ==========================================
# This dictionary matches the structure of your Crop Yield Training Data
# Used for the "Data Guide" display in the app.
DATA_EXAMPLE = {
    'ID': {0: 'ID_GTFAC7PEVWQ9', 1: 'ID_TK40ARLSPOKS', 2: 'ID_1FJY2CRIMLZZ'},
    'District': {0: 'Nalanda', 1: 'Nalanda', 2: 'Gaya'},
    'Block': {0: 'Noorsarai', 1: 'Rajgir', 2: 'Gurua'},
    'CultLand': {0: 45, 1: 26, 2: 10},
    'CropCultLand': {0: 40, 1: 26, 2: 10},
    'LandPreparationMethod': {
        0: 'TractorPlough FourWheelTracRotavator',
        1: 'WetTillagePuddling TractorPlough FourWheelTracRotavator',
        2: 'TractorPlough FourWheelTracRotavator'
    },
    'CropTillageDate': {0: '2022-07-20', 1: '2022-07-18', 2: '2022-06-30'},
    'CropTillageDepth': {0: 5, 1: 5, 2: 6},
    'CropEstMethod': {0: 'Manual_PuddledRandom', 1: 'Manual_PuddledRandom', 2: 'Manual_PuddledRandom'},
    'RcNursEstDate': {0: '2022-06-27', 1: '2022-06-20', 2: '2022-06-20'},
    'SeedingSowingTransplanting': {0: '2022-07-21', 1: '2022-07-20', 2: '2022-08-13'},
    'SeedlingsPerPit': {0: 2.0, 1: 2.0, 2: 2.0},
    'NursDetFactor': {
        0: 'CalendarDate IrrigWaterAvailability SeedAvailability',
        1: 'CalendarDate PreMonsoonShowers IrrigWaterAvailability LabourAvailability SeedAvailability',
        2: 'PreMonsoonShowers IrrigWaterAvailability LabourAvailability'
    },
    'TransDetFactor': {
        0: 'CalendarDate SeedlingAge RainArrival IrrigWaterAvailability LaborAvailability',
        1: 'CalendarDate SeedlingAge RainArrival IrrigWaterAvailability LaborAvailability',
        2: 'SeedlingAge IrrigWaterAvailability LaborAvailability'
    },
    'TransplantingIrrigationHours': {0: 5.0, 1: 5.0, 2: 4.0},
    'TransplantingIrrigationSource': {0: 'Boring', 1: 'Boring', 2: 'Boring'},
    'TransplantingIrrigationPowerSource': {0: 'Electric', 1: 'Electric', 2: 'Electric'},
    'TransIrriCost': {0: 200.0, 1: 125.0, 2: 80.0},
    'StandingWater': {0: 2.0, 1: 3.0, 2: 2.0},
    'OrgFertilizers': {0: np.nan, 1: np.nan, 2: 'Ganaura FYM'},
    'Ganaura': {0: np.nan, 1: np.nan, 2: 1.0},
    'CropOrgFYM': {0: np.nan, 1: np.nan, 2: 1.0},
    'PCropSolidOrgFertAppMethod': {0: np.nan, 1: np.nan, 2: 'SoilApplied'},
    'NoFertilizerAppln': {0: 2, 1: 2, 2: 2},
    'CropbasalFerts': {0: 'Urea', 1: 'DAP Urea', 2: 'DAP'},
    'BasalDAP': {0: np.nan, 1: 15.0, 2: 4.0},
    'BasalUrea': {0: 20.0, 1: 10.0, 2: np.nan},
    'MineralFertAppMethod': {0: 'Broadcasting', 1: 'Broadcasting', 2: 'SoilApplied'},
    'FirstTopDressFert': {0: 'Urea', 1: 'Urea', 2: 'Urea'},
    '1tdUrea': {0: 15.0, 1: 20.0, 2: 5.0},
    '1appDaysUrea': {0: 18.0, 1: 39.0, 2: 65.0},
    '2tdUrea': {0: np.nan, 1: np.nan, 2: np.nan},
    '2appDaysUrea': {0: np.nan, 1: np.nan, 2: np.nan},
    'MineralFertAppMethod.1': {0: 'Broadcasting', 1: 'Broadcasting', 2: 'RootApplication'},
    'Harv_method': {0: 'machine', 1: 'hand', 2: 'hand'},
    'Harv_date': {0: '2022-11-16', 1: '2022-11-25', 2: '2022-12-12'},
    'Harv_hand_rent': {0: np.nan, 1: 3.0, 2: 480.0},
    'Threshing_date': {0: '2022-11-16', 1: '2022-12-24', 2: '2023-01-11'},
    'Threshing_method': {0: 'machine', 1: 'machine', 2: 'machine'},
    'Residue_length': {0: 30, 1: 24, 2: 30},
    'Residue_perc': {0: 40, 1: 10, 2: 10},
    'Stubble_use': {0: 'plowed_in_soil', 1: 'plowed_in_soil', 2: 'plowed_in_soil'},
    'Acre': {0: 0.3125, 1: 0.3125, 2: 0.1481481481481481}
}

# ==========================================
# 2. STREAMLIT CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AgriYield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main() -> None:
    """
    Main execution function for the Streamlit application.
    Handles UI layout, File Ingestion, and Model Inference execution.
    """
    
    # --- SIDEBAR ---
    with st.sidebar:
        # Note: Ensure 'assets/logo.png' exists, or remove this line
        try:
            st.image("assets/app_logo.png", use_container_width=True) 
        except:
            st.warning("Logo not found at assets/app_logo.png")
            
        st.title("⚙️ System Control")
        st.info(f"**Current Model:** {Config.MODEL_TYPE.upper()}")
        
        st.write("---")
        st.markdown("""
        ### 🧪 Methodology
        This pipeline leverages **Agritech Data** to forecast yield:
        * 🚜 **Land Prep:** Tillage depth & methods.
        * 🗓️ **Phenology:** Sowing & Harvest dates.
        * 💧 **Irrigation:** Water source & timing.
        * 🧪 **Fertilizers:** Urea/DAP application usage.
        """)
        
        st.write("---")
        st.caption(f"Experiment: `{Config.EXPERIMENT_NAME}`")

    # --- HEADER ---
    st.title("🌾 AgriYield: Crop Production Forecasting")
    st.markdown("""
    **Optimize your harvest.** 🚜  
    Upload your farm survey data below to generate precision yield estimates using our advanced Machine Learning pipeline.
    """)

    # --- SECTION 1: DATA GUIDE ---
    with st.expander("📖 View Required Data Format (CSV Structure)"):
        st.write("To ensure accurate predictions, your CSV must match this schema:")
        st.dataframe(pd.DataFrame(DATA_EXAMPLE).head(3), use_container_width=True)
        st.caption("Note: Ensure date columns (e.g., CropTillageDate) are formatted correctly (YYYY-MM-DD).")

    st.divider()

    # --- SECTION 2: INFERENCE ---
    st.subheader("📤 Step 1: Upload Farm Data")
    
    uploaded_file = st.file_uploader(
        "Drop your Survey CSV file here", 
        type="csv", 
        help="Upload the dataset containing District, Block, CultLand, and farming practice columns."
    )

    if uploaded_file is not None:
        col_preview, col_action = st.columns([2, 1])
        
        with col_preview:
            # Read and display
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File `{uploaded_file.name}` uploaded successfully!")
            st.write(f"🔍 **Data Preview** ({df.shape[0]} rows):")
            st.dataframe(df.head(5), height=200, use_container_width=True)

        with col_action:
            st.write("🚀 **Action Zone**")
            
            # Predict Button
            if st.button("✨ Generate Yield Estimates", type="primary", use_container_width=True):
                with st.status("🤖 Analyzing agricultural patterns...", expanded=True) as status:
                    
                    # 1. Save temporary file for the pipeline to read
                    file_name = "user_input.csv"
                    file_dir = Config.TEST_DATA_PATH.parent / file_name
                    
                    # Ensure directory exists
                    if not Config.TEST_DATA_PATH.parent.exists():
                        Config.TEST_DATA_PATH.parent.mkdir(parents=True)
                        
                    df.to_csv(file_dir, index=False)
                    st.write("📁 Input data staged.")
                    
                    # 2. Run Inference Pipeline
                    st.write(f"🚜 Running `{Config.MODEL_TYPE}` regressor...")
                    try:
                        make_predictions(str(file_dir))
                        st.write("📊 Aggregating results...")
                        status.update(label="✅ Prediction Complete!", state="complete", expanded=False)
                        
                    except Exception as e:
                        status.update(label="❌ Error during inference", state="error")
                        st.error(f"Pipeline failed: {e}")
                        st.stop()

                # 3. Download Logic
                output_path = Config.DATA_DIR / "outputs" / "submission.csv"
                if output_path.exists():
                    submission = pd.read_csv(output_path)
                    csv_data = submission.to_csv(index=False).encode('utf-8')
                    
                    st.success("Analysis ready for download.")
                    st.download_button(
                        label="📥 Download Yield_Predictions.csv",
                        data=csv_data,
                        file_name="agriyield_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.error("Prediction output file not found.")

    st.divider()

    # --- SECTION 3: VISUALS ---
    st.subheader("📊 Step 2: Model Explainability")
    st.markdown("Understanding the drivers behind the yield predictions.")

    # Create the side-by-side layout
    img_col1, img_col2 = st.columns(2)
    
    with img_col1:
        st.markdown("#### 🏆 Feature Importance")
        importance_path = Config.FIGURES_DIR / f"{Config.MODEL_TYPE}_importance.png"
        
        if importance_path.exists():
            st.image(str(importance_path), caption="Top factors driving crop yield (e.g., Acreage, Fertilizers)", use_container_width=True)
        else:
            st.info("Feature importance plot is available after model training.")

    with img_col2:
        st.markdown("#### 📈 Prediction Accuracy")
        fit_path = Config.FIGURES_DIR / "regression_fit.png"
        
        if fit_path.exists():
            st.image(str(fit_path), caption="Actual Yield vs. Predicted Yield Validation", use_container_width=True)
        else:
            st.info("Regression fit plot is available after model training.")

if __name__ == "__main__":
    main()