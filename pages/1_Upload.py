import streamlit as st
import pandas as pd
import zipfile
import os
from functools import reduce

st.set_page_config(page_title="Upload Dataset", page_icon="📥", layout="centered")

st.title("📥 Upload Dataset")

# if Already Uploaded
if "df" in st.session_state:
    st.success("✔ Dataset already uploaded")
    st.write(f"**Dataset Name:** {st.session_state['dataset']}")
    st.dataframe(st.session_state["df"].head())
    st.stop()

# convert one file
def load_dataframe(file):
    try:
        return pd.read_csv(file)
    except Exception:
        try:
            return pd.read_excel(file)
        except Exception:
            st.error("❌ Unsupported file format.")
            st.stop()
# convert and combine multiple files
def merge_multiple_files(folder_path, file_list):
    dataframes = []

    for file_name in file_list:
        if file_name.endswith((".csv", ".xls", ".xlsx", ".xlsb", ".xlsm", ".ods")):
            df = load_dataframe(os.path.join(folder_path, file_name))
            dataframes.append(df)

    if not dataframes:
        st.error("❌ No valid datasets found inside ZIP.")
        st.stop()

    common_columns = set(dataframes[0].columns)
    for df in dataframes[1:]:
        common_columns = common_columns.intersection(df.columns)

    if not common_columns:
        st.error("❌ No common columns found to merge datasets.")
        st.stop()

    common_columns = list(common_columns)

    st.info(f"🔗 Merging datasets on common column: {common_columns[0]}")

    merged_df = reduce(
        lambda df1, df2: pd.merge(df1, df2, on=common_columns, how="outer"),
        dataframes
    )

    return merged_df


# File Upload
uploaded_file = st.file_uploader(
    "Upload CSV / Excel file or ZIP file",
    type=["csv", "xls", "xlsx", "xlsb", "xlsm", "ods", "zip"]
)

if uploaded_file:

    if uploaded_file.name.endswith(".zip"):

        temp_folder = "temp_data"
        os.makedirs(temp_folder, exist_ok=True)

        with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
            zip_ref.extractall(temp_folder)

        files = os.listdir(temp_folder)

        with st.spinner("📚 Merging multiple datasets..."):
            df = merge_multiple_files(temp_folder, files)

    else:
        df = load_dataframe(uploaded_file)

    # Dataset Size Safety (Cloud Limit)
    if df.shape[0] > 50000:
        st.warning("⚠ Dataset too large for Streamlit free deployment. Please upload smaller dataset (<50,000 rows).")
        st.stop()

    st.session_state["df"] = df
    st.session_state["dataset"] = uploaded_file.name.split(".")[0]

    st.success("✅ Dataset uploaded successfully!")
    st.write("### Preview")
    st.dataframe(df.head())
