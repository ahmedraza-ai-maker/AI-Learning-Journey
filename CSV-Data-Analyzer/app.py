import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="CSV Data Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 CSV Data Analyzer")
st.write("Upload a CSV file and analyze your dataset instantly.")


uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)


if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("CSV file uploaded successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(df)

    st.subheader("Dataset Information")

    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])

    st.subheader("Data Visualization")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if len(numeric_columns) > 0:

        selected_column = st.selectbox(
            "Select a numeric column",
            numeric_columns
        )

        fig, ax = plt.subplots()

        ax.hist(
            df[selected_column],
            bins=10
        )

        ax.set_title(
            f"Distribution of {selected_column}"
        )

        ax.set_xlabel(selected_column)
        ax.set_ylabel("Frequency")

        st.pyplot(fig)