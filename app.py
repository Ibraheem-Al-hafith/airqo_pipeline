import streamlit as st
import pandas as pd
from src.config import Config
from src.inference.predict import make_predictions


def main() -> None:
    # 1. Basic UI: Title and Description

    st.title("AirQo African Air Quality Prediction")
    st.markdown("""
    ### Problem Description:
                this is a place holder for the problem description and 
                what this web app does.
        """)
    
    # 2. File Uploading:
    uploaded_file = st.file_uploader("Choose a CSV file", type= "csv")
    if uploaded_file is not None:
        # Read the file:
        file_name = "user.csv"
        file_dir = Config.TEST_DATA_PATH.parent / file_name
        df = pd.read_csv(uploaded_file)

        # Show a preview of the data
        st.write("### Data Preview")
        st.dataframe(df.head(2))

        # save the data because make_predictions expect a file path
        df.to_csv(file_dir, index=False)

        # Get the predicions:
        make_predictions(str(file_dir))

        # get the submission dataframe:
        submission = pd.read_csv(Config.DATA_DIR / "outputs" / "submission.csv")

        # convert the data to downloadble file which streamlit can handle it
        csv_data = submission.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions",
            data=csv_data,
            file_name="sample_submission.csv",
            mime="text/csv"
        )
    
    # 3. Display Figures :
    #st.write("### Model Insights")
    #st.image("path to feature importance.png", caption="Top 5 Feature Importances")


if __name__=="__main__":
    main()
