import streamlit as st

st.set_page_config(
    page_title="MVP Predict Program",
    page_icon="🎓",
    layout="wide"
)

if 'student_df' not in st.session_state:
    st.session_state['student_df'] = None

home = st.Page("pages/0_main_page.py", title="🏠 Главная", default=True)
page_1 = st.Page("pages/1_upload_data.py", title="📂 Загрузка данных")
page_2 = st.Page("pages/2_predict.py", title="🔮 Прогнозирование")
page_3 = st.Page("pages/3_global_analyse.py", title="📊 Глобальная аналитика")

pg = st.navigation([home, page_1, page_2, page_3])

pg.run()