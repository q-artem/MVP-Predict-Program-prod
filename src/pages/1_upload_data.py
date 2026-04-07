import streamlit as st
import pandas as pd

st.set_page_config(page_title="Загрузка данных", page_icon="📁", layout="wide")

st.title("📁 Загрузка ведомости студентов")
st.markdown(
    "Пожалуйста, загрузите файл формата **.csv**. Убедитесь, что заголовки столбцов соответствуют шаблону.")

uploaded_file = st.file_uploader("Перетащите файл сюда или нажмите 'Browse files'", type=["csv"])

def process_file():
    try:
        df = st.session_state.get('student_df')

        st.subheader("Предварительный просмотр данных:")
        st.write("Первые 10 строк загруженного файла:")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Статистика успеваемости")
        c1, c2, c3 = st.columns(3)
        c1.metric("Всего записей", len(df))
        quality_score = "ОЦЕНКА_NUM"
        binary_score = "Оценка (0 или 1)"
        if quality_score in df.columns:
            c2.metric("Средний GPA по вузу", f"{df[quality_score].mean():.2f}")
        else:
            c2.metric("Процент зачётов по вузу", f"{df[binary_score].mean() * 100:.2f} %")

        if "Несдано_вовремя_сем_1" in df.columns:
            arrears_col_name = "Несдано_вовремя_сем_"
        else:
            arrears_col_name = "Несдано_бинарных_вовремя_сем_"
        c3.metric("Макс. задолженностей", int(max([df[f"{arrears_col_name}{q}"].max() if f"{arrears_col_name}{q}" in df.columns else 0 for q in range(9)])))

        with st.expander("Посмотреть статистику по пропускам (NaN)"):
            missing_data = df.isna().sum()
            missing_data = missing_data[missing_data > 0]
            if not missing_data.empty:
                st.warning("В загруженных данных есть пропуски. Они будут обработаны моделью автоматически. Качество предсказаний может немного снизиться")
                st.dataframe(pd.DataFrame({'Пропущено записей': missing_data}))
            else:
                st.info("Пропусков в данных не обнаружено.")

        st.info("Данные сохранены в памяти. Теперь вы можете перейти на вкладку **«Прогнозирование»** в левом меню.")

    except Exception as e:
        st.error(f"❌ Ошибка при предварительном просмотре: {e}")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, engine="pyarrow")

        # Если успешно, сохраняем в датафрейм
        st.session_state['student_df'] = df

        st.success(f"✅ Файл успешно загружен! Распознано {len(df)} записей (студентов).")
    except Exception as e:
        st.error(f"❌ Ошибка при чтении файла: {e}")
        st.session_state['student_df'] = None

    process_file()


# Если перезашли
elif st.session_state.get('student_df') is not None:
    st.info("В памяти уже находится загруженный ранее файл. Вы можете загрузить новый или продолжить работу со старым. Ниже представлена информация на основе старого файла.")

    process_file()
