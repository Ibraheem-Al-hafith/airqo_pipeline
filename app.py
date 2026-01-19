import streamlit as st
import pandas as pd
import numpy as np
from src.config import Config
from src.inference.predict import make_predictions

data_example = {'id': {0: 'id_vjcx08sz91', 1: 'id_bkg215syli', 2: 'id_oui2pot3qd'},
 'site_id': {0: '6531a46a89b3300013914a36',
  1: '6531a46a89b3300013914a36',
  2: '6531a46a89b3300013914a36'},
 'site_latitude': {0: 6.53257, 1: 6.53257, 2: 6.53257},
 'site_longitude': {0: 3.39936, 1: 3.39936, 2: 3.39936},
 'city': {0: 'Lagos', 1: 'Lagos', 2: 'Lagos'},
 'country': {0: 'Nigeria', 1: 'Nigeria', 2: 'Nigeria'},
 'date': {0: '2023-10-25', 1: '2023-11-02', 2: '2023-11-03'},
 'hour': {0: 13, 1: 12, 2: 13},
 'sulphurdioxide_so2_column_number_density': {0: np.nan, 1: np.nan, 2: np.nan},
 'sulphurdioxide_so2_column_number_density_amf': {0: np.nan, 1: np.nan, 2: np.nan},
 'sulphurdioxide_so2_slant_column_number_density': {0: np.nan, 1: np.nan, 2: np.nan},
 'sulphurdioxide_cloud_fraction': {0: np.nan, 1: np.nan, 2: np.nan},
 'sulphurdioxide_sensor_azimuth_angle': {0: np.nan, 1: np.nan, 2: np.nan},
 'sulphurdioxide_sensor_zenith_angle': {0: np.nan, 1: np.nan, 2: np.nan},
 'sulphurdioxide_solar_azimuth_angle': {0: np.nan, 1: np.nan, 2: np.nan},
 'sulphurdioxide_solar_zenith_angle': {0: np.nan, 1: np.nan, 2: np.nan},
 'sulphurdioxide_so2_column_number_density_15km': {0: np.nan, 1: np.nan, 2: np.nan},
 'month': {0: 10.0, 1: 11.0, 2: 11.0},
 'carbonmonoxide_co_column_number_density': {0: np.nan,
  1: 0.0454752769339527,
  2: np.nan},
 'carbonmonoxide_h2o_column_number_density': {0: np.nan,
  1: 3771.027210467723,
  2: np.nan},
 'carbonmonoxide_cloud_height': {0: np.nan, 1: 3399.75684527907, 2: np.nan},
 'carbonmonoxide_sensor_altitude': {0: np.nan, 1: 828569.6238062604, 2: np.nan},
 'carbonmonoxide_sensor_azimuth_angle': {0: np.nan, 1: 69.24535082958508, 2: np.nan},
 'carbonmonoxide_sensor_zenith_angle': {0: np.nan, 1: 59.1596946834736, 2: np.nan},
 'carbonmonoxide_solar_azimuth_angle': {0: np.nan,
  1: -143.37057538316276,
  2: np.nan},
 'carbonmonoxide_solar_zenith_angle': {0: np.nan, 1: 26.566997473264017, 2: np.nan},
 'nitrogendioxide_no2_column_number_density': {0: np.nan, 1: np.nan, 2: np.nan},
 'nitrogendioxide_tropospheric_no2_column_number_density': {0: np.nan,
  1: np.nan,
  2: np.nan},
 'nitrogendioxide_stratospheric_no2_column_number_density': {0: np.nan,
  1: np.nan,
  2: np.nan},
 'nitrogendioxide_no2_slant_column_number_density': {0: np.nan, 1: np.nan, 2: np.nan},
 'nitrogendioxide_tropopause_pressure': {0: np.nan, 1: np.nan, 2: np.nan},
 'nitrogendioxide_absorbing_aerosol_index': {0: np.nan, 1: np.nan, 2: np.nan},
 'nitrogendioxide_cloud_fraction': {0: np.nan, 1: np.nan, 2: np.nan},
 'nitrogendioxide_sensor_altitude': {0: np.nan, 1: np.nan, 2: np.nan},
 'nitrogendioxide_sensor_azimuth_angle': {0: np.nan, 1: np.nan, 2: np.nan},
 'nitrogendioxide_sensor_zenith_angle': {0: np.nan, 1: np.nan, 2: np.nan},
 'nitrogendioxide_solar_azimuth_angle': {0: np.nan, 1: np.nan, 2: np.nan},
 'nitrogendioxide_solar_zenith_angle': {0: np.nan, 1: np.nan, 2: np.nan},
 'formaldehyde_tropospheric_hcho_column_number_density': {0: np.nan,
  1: 0.000214044630411,
  2: np.nan},
 'formaldehyde_tropospheric_hcho_column_number_density_amf': {0: np.nan,
  1: 1.4623903036117585,
  2: np.nan},
 'formaldehyde_hcho_slant_column_number_density': {0: np.nan,
  1: 0.0002400214143563,
  2: np.nan},
 'formaldehyde_cloud_fraction': {0: np.nan, 1: 0.359149873256684, 2: np.nan},
 'formaldehyde_solar_zenith_angle': {0: np.nan, 1: 26.525512695312504, 2: np.nan},
 'formaldehyde_solar_azimuth_angle': {0: np.nan, 1: -143.48016357421875, 2: np.nan},
 'formaldehyde_sensor_zenith_angle': {0: np.nan, 1: 59.220096588134766, 2: np.nan},
 'formaldehyde_sensor_azimuth_angle': {0: np.nan, 1: 70.8759536743164, 2: np.nan},
 'uvaerosolindex_absorbing_aerosol_index': {0: 0.0523008033633217,
  1: -0.3152063488960272,
  2: 1.0978158712387107},
 'uvaerosolindex_sensor_altitude': {0: 828817.9374999763,
  1: 828578.6250000016,
  2: 828878.6875000016},
 'uvaerosolindex_sensor_azimuth_angle': {0: -100.80514526367188,
  1: 70.8759536743164,
  2: -96.41194152832033},
 'uvaerosolindex_sensor_zenith_angle': {0: 21.720518112182617,
  1: 59.220096588134766,
  2: 61.04500961303711},
 'uvaerosolindex_solar_azimuth_angle': {0: -123.52379608154298,
  1: -143.48016357421875,
  2: -121.30712127685548},
 'uvaerosolindex_solar_zenith_angle': {0: 33.74591445922852,
  1: 26.525512695312504,
  2: 41.89811325073242},
 'ozone_o3_column_number_density': {0: 0.1220549121499026,
  1: 0.116974785923958,
  2: 0.1175593361258509},
 'ozone_o3_column_number_density_amf': {0: 2.3014037609099685,
  1: 3.049901723861701,
  2: 3.248702764511115},
 'ozone_o3_slant_column_number_density': {0: 0.2858027815818705,
  1: 0.3622033298015601,
  2: 0.3841677904129036},
 'ozone_o3_effective_temperature': {0: 230.69375610350903,
  1: 228.2601928710942,
  2: 224.10246276855517},
 'ozone_cloud_fraction': {0: 0.9060392379760482,
  1: 0.3647132515907295,
  2: 0.7541627883911148},
 'ozone_sensor_azimuth_angle': {0: -100.80514526367188,
  1: 70.8759536743164,
  2: -96.41194152832033},
 'ozone_sensor_zenith_angle': {0: 21.720518112182617,
  1: 59.220096588134766,
  2: 61.04500961303711},
 'ozone_solar_azimuth_angle': {0: -123.52379608154298,
  1: -143.48016357421875,
  2: -121.30712127685548},
 'ozone_solar_zenith_angle': {0: 33.74591445922852,
  1: 26.525512695312504,
  2: 41.89811325073242},
 'uvaerosollayerheight_aerosol_height': {0: np.nan, 1: np.nan, 2: np.nan},
 'uvaerosollayerheight_aerosol_pressure': {0: np.nan, 1: np.nan, 2: np.nan},
 'uvaerosollayerheight_aerosol_optical_depth': {0: np.nan, 1: np.nan, 2: np.nan},
 'uvaerosollayerheight_sensor_zenith_angle': {0: np.nan, 1: np.nan, 2: np.nan},
 'uvaerosollayerheight_sensor_azimuth_angle': {0: np.nan, 1: np.nan, 2: np.nan},
 'uvaerosollayerheight_solar_azimuth_angle': {0: np.nan, 1: np.nan, 2: np.nan},
 'uvaerosollayerheight_solar_zenith_angle': {0: np.nan, 1: np.nan, 2: np.nan},
 'cloud_cloud_fraction': {0: np.nan, 1: np.nan, 2: 0.7563918871573639},
 'cloud_cloud_top_pressure': {0: np.nan, 1: np.nan, 2: 45185.499589698775},
 'cloud_cloud_top_height': {0: np.nan, 1: np.nan, 2: 6791.68288790298},
 'cloud_cloud_base_pressure': {0: np.nan, 1: np.nan, 2: 51171.80248564955},
 'cloud_cloud_base_height': {0: np.nan, 1: np.nan, 2: 5791.682829397136},
 'cloud_cloud_optical_depth': {0: np.nan, 1: np.nan, 2: 11.816715046824454},
 'cloud_surface_albedo': {0: np.nan, 1: np.nan, 2: 0.1927570952348716},
 'cloud_sensor_azimuth_angle': {0: np.nan, 1: np.nan, 2: -96.41188990612474},
 'cloud_sensor_zenith_angle': {0: np.nan, 1: np.nan, 2: 61.04512277641219},
 'cloud_solar_azimuth_angle': {0: np.nan, 1: np.nan, 2: -121.30741441488706},
 'cloud_solar_zenith_angle': {0: np.nan, 1: np.nan, 2: 41.89826925713596}
 }

# Page configuration for a professional look
st.set_page_config(
    page_title="AirQo Predictor",
    page_icon="🌍",
    layout="wide"
)


def main() -> None:
    # --- SIDEBAR ---
    with st.sidebar:
        st.image("https://airqo.africa/static/img/airqo-logo.png", width=150) # Assuming AirQo logo URL
        st.title("⚙️ Settings & Info")
        st.info(f"**Current Model:** {Config.MODEL.upper()}")
        st.write("---")
        st.markdown("""
        ### 🧪 Methodology
        This pipeline uses satellite observations (AOD) to estimate ground-level air quality.
        """)

    # --- HEADER ---
    st.title("🌍 AirQo African Air Quality Prediction")
    st.markdown("""
    Welcome to the **AirQo Inference Portal**! 💨  
    This tool allows you to upload satellite data and generate high-accuracy air quality predictions in seconds.
    """)

    # --- SECTION 1: DATA GUIDE ---
    with st.expander("📖 View Required Data Format (CSV Structure)"):
        st.write("Your uploaded file must match this structure:")
        st.dataframe(pd.DataFrame(data_example).head(3))
        st.caption("Note: Ensure all satellite density columns are present.")

    st.divider()

    # --- SECTION 2: INFERENCE ---
    st.subheader("📤 Step 1: Upload your Data")
    uploaded_file = st.file_uploader("Drop your CSV file here", type="csv", help="Upload the satellite observation features.")

    if uploaded_file is not None:
        col_preview, col_action = st.columns([2, 1])
        
        with col_preview:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File uploaded successfully!")
            st.write("🔍 **Data Preview:**")
            st.dataframe(df.head(5), height=200)

        with col_action:
            st.write("🚀 **Action Zone**")
            if st.button("✨ Generate Predictions"):
                with st.status("🤖 AI is processing data...", expanded=True) as status:
                    # Save temporary file
                    file_name = "user_input.csv"
                    file_dir = Config.TEST_DATA_PATH.parent / file_name
                    df.to_csv(file_dir, index=False)
                    
                    # Run Pipeline
                    st.write("🔗 Loading Model Pipeline...")
                    make_predictions(str(file_dir))
                    st.write("📊 Finalizing submission format...")
                    
                    status.update(label="✅ Prediction Complete!", state="complete", expanded=False)

                # Download Logic
                submission = pd.read_csv(Config.DATA_DIR / "outputs" / "submission.csv")
                csv_data = submission.to_csv(index=False).encode('utf-8')
                
                st.balloons() # Celebration!
                st.download_button(
                    label="📥 Download Predictions.csv",
                    data=csv_data,
                    file_name="airqo_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    st.divider()

    # --- SECTION 3: VISUALS ---
    st.subheader("📊 Step 2: Model Performance & Insights")
    
    # Create the side-by-side layout
    img_col1, img_col2 = st.columns(2)
    
    with img_col1:
        st.markdown("#### 🏆 Feature Importance")
        importance_path = Config.FIGURES_DIR / f"{Config.MODEL}_importance.png"
        if importance_path.exists():
            st.image(str(importance_path), caption="Which features influenced the model most?", use_container_width=True)
        else:
            st.warning("Feature importance plot not found.")

    with img_col2:
        st.markdown("#### 📈 Regression Fit")
        fit_path = Config.FIGURES_DIR / "regression_fit.png"
        if fit_path.exists():
            st.image(str(fit_path), caption="Actual vs. Predicted Values", use_container_width=True)
        else:
            st.warning("Regression fit plot not found.")

if __name__ == "__main__":
    main()