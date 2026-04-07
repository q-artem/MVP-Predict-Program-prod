import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import math

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_utils import get_paths, load_model, prepare_features

# НАСТРОЙКИ СТРАНИЦЫ
st.set_page_config(page_title="Прогнозирование", page_icon="🔮", layout="wide")

st.title("🔮 Модуль прогнозирования и аналитики")

# Проверка: загрузил ли пользователь данные на Шаге 1
if 'student_df' not in st.session_state or st.session_state['student_df'] is None:
    st.warning("⚠️ Данные не найдены. Пожалуйста, вернитесь на вкладку «Загрузка данных» и загрузите файл.")
    st.stop()

df = st.session_state['student_df']


# ЗАГРУЗКА МОДЕЛИ
@st.cache_resource
def get_cached_model(model_type):
    """Загружает модель 1 раз и держит в памяти"""
    paths = get_paths(model_type)
    model, columns = load_model(paths)
    return model, columns


# Выбор задачи
st.subheader("1. Выберите тип прогноза")
task_choice = st.radio(
    "Что предсказываем?",
    options=["Сдача предмета (Сдаст / Не сдаст)", "Итоговая оценка (2, 3, 4 или 5)"],
    horizontal=True,
    label_visibility="collapsed"
)
model_type = 0 if "Сдача" in task_choice else 1

# Загружаем выбранную модель
try:
    model, saved_columns = get_cached_model(model_type)
except Exception as e:
    st.error(f"❌ Ошибка загрузки модели. Проверьте папку models. Детали: {e}")
    st.stop()

# ДВЕ ВКЛАДКИ
st.divider()
tab1, tab2, tab3 = st.tabs(["Индивидуальный профиль", "Массовый расчёт", "Ручной ввод"])

st.markdown("""
    <style>
    /* Находим кнопки вкладок и меняем шрифт внутри них */
    button[data-baseweb="tab"] div p {
        font-size: 18px !important; /* Размер текста */
        font-weight: 400 !important; /* Жирность (800 - очень жирный) */
        font-family: "Source Sans Pro", sans-serif;
    }

    /* Увеличиваем высоту и отступы вкладок, чтобы они выглядели массивнее */
    button[data-baseweb="tab"] {
        height: 50px !important;
        padding-left: 20px !important;
        padding-right: 20px !important;
        border-radius: 10px 10px 0px 0px; /* Скругление углов сверху */
    }

    /* Стиль для активной (выбранной) вкладки */
    button[data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(255, 75, 75, 0.1) !important; /* Легкий фон под активной вкладкой */
        border-bottom: 3px solid rgb(255, 75, 75) !important; /* Жирная линия снизу */
    }
    </style>
""", unsafe_allow_html=True)

# ВКЛАДКА 1: ИНДИВИДУАЛЬНЫЙ ПРОФИЛЬ
with tab1:
    st.subheader("2. Выберите студента, семестр и дисциплину")

    id_col = 'ИД'
    sem_col = 'СЕМЕСТР'
    disc_col = 'ДИСЦИПЛИНА'

    col_sel1, col_sel2, col_sel3 = st.columns(3)

    # Списки
    with col_sel1:
        student_id = st.selectbox("1. Выберите ID студента", sorted(df[id_col].unique()))

    student_all_records = df[df[id_col] == student_id]

    with col_sel2:
        available_sems = sorted(student_all_records[sem_col].unique())
        default_sem_index = len(available_sems) - 1

        selected_sem = st.selectbox("2. Выберите семестр", available_sems, index=default_sem_index)

    student_sem_records = student_all_records[student_all_records[sem_col] == selected_sem]

    with col_sel3:
        available_disciplines = sorted(student_sem_records[disc_col].unique())
        selected_disc = st.selectbox("3. Выберите дисциплину", available_disciplines)

    st.divider()

    st.subheader("3. Общая информация")

    student_raw_row = student_sem_records[student_sem_records[disc_col] == selected_disc].head(1)

    row_dict = student_raw_row.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    current_gpa = row_dict.get(f"Средняя_оценка_кач_сем_{int(selected_sem) - 1}",
                               row_dict.get(f"Средняя_оценка_сем_{int(selected_sem) - 1}", None))
    debts = row_dict.get(f"Несдано_бинарных_вовремя_сем_{int(selected_sem) - 1}",
                         row_dict.get(f"Несдано_вовремя_сем_{int(selected_sem) - 1}", None))
    payment = row_dict.get('УСЛОВИЕ_ЗАЧИСЛЕНИЯ', 'Не указано')

    c1.metric("Текущий GPA (до сдачи)", f"{round(current_gpa, 2) if current_gpa is not None else "Не определён"}")
    c2.metric("Накоплено долгов", f"{int(debts) if debts is not None else "Не определено"}")
    c3.metric("Анализируемый семестр", int(selected_sem))
    c4.metric("Форма обучения", str(payment))

    st.divider()

    col_left, col_right = st.columns([1.2, 0.8])

    with col_left:
        st.subheader("4. Прогноз и интерпретация")
        with st.spinner('Анализируем факторы...'):
            try:
                X_student = prepare_features(student_raw_row, saved_columns=saved_columns)

                # Делаем предсказание
                pred = model.predict(X_student)[0]
                pred_text = "Сдаст 🟢" if (
                        model_type == 0 and pred == 1) else "Не сдаст 🔴" if model_type == 0 else f"Оценка: {pred}"
                st.success(f"##### **Прогноз системы по предмету «{selected_disc}»:** {pred_text}")

                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_student)

                if model_type == 0:
                    sample_vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
                    base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (list,
                                                                                                    np.ndarray)) else explainer.expected_value
                else:
                    cls_idx = list(model.classes_).index(pred)
                    sample_vals = shap_values[cls_idx][0] if isinstance(shap_values, list) else shap_values[
                        0, :, cls_idx]
                    base_val = explainer.expected_value[cls_idx] if isinstance(explainer.expected_value, (list,
                                                                                                          np.ndarray)) else explainer.expected_value

                fig, ax = plt.subplots(figsize=(10, 6))
                exp = shap.Explanation(
                    values=sample_vals,
                    base_values=base_val,
                    data=X_student.iloc[0].values,
                    feature_names=X_student.columns.tolist()
                )

                shap.plots.waterfall(exp, show=False, max_display=10)

                plt.tight_layout()
                st.pyplot(fig, bbox_inches='tight')
                plt.close(fig)

            except Exception as e:
                st.error(f"Не удалось рассчитать логику модели: {e}")
                st.info("Попробуйте выбрать другой тип прогноза")

    with col_right:
        st.subheader("5. Сравнение с потоком")

        col_score = "ОЦЕНКА_NUM" if "ОЦЕНКА_NUM" in df.columns else "Оценка (0 или 1)"

        avg_disc_gpa = df[df[disc_col] == selected_disc][col_score].mean()
        avg_debts_col = f"Несдано_бинарных_вовремя_сем_{int(selected_sem) - 1}" if f"Несдано_бинарных_вовремя_сем_{int(selected_sem) - 1}" in df.columns else f"Несдано_вовремя_сем_{int(selected_sem) - 1}"
        avg_debts = df[df[disc_col] == selected_disc][avg_debts_col].mean() if int(selected_sem) > 1 else 0

        if pd.isna(avg_disc_gpa):
            st.info("Нет достаточных исторических данных для этого предмета.")
        else:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pred if model_type == 1 else current_gpa,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Прогноз относительно средней<br>оценки по предмету", 'font': {'size': 14}},
                delta={'reference': avg_disc_gpa, 'increasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [2, 5], 'tickwidth': 1},
                    'bar': {'color': "royalblue"},
                    'steps': [
                        {'range': [2, 3], 'color': '#ffcccc'},
                        {'range': [3, 4], 'color': '#fff0b3'},
                        {'range': [4, 5], 'color': '#ccffcc'}
                    ],
                    'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': avg_disc_gpa}
                }
            ))
            fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.divider()

        st.subheader("6. Профиль компетенций")
        try:
            val_gpa = (current_gpa - 2) / 3 if current_gpa > 2 else 0
            val_discipline = max(0, 1 - (debts / 5))
            val_experience = selected_sem / 8
            val_stability = 1.0 if debts == 0 else 0.5

            categories = ['Успеваемость (GPA)', 'Дисциплина (Нет долгов)',
                          'Опыт (Семестр)', 'Стабильность (Попытки сдачи)']

            avg_val_gpa = (avg_disc_gpa - 2) / 3
            avg_val_discipline = max(0, 1 - (avg_debts / 5))
            avg_val_experience = df[sem_col].mean() / 8
            avg_val_stability = 0.7  # Эталонное значение

            fig_radar = go.Figure()

            # Слой среднего по вузу
            fig_radar.add_trace(go.Scatterpolar(
                r=[avg_val_gpa, avg_val_discipline, avg_val_experience, avg_val_stability, avg_val_gpa],
                theta=categories + [categories[0]],
                fill='toself',
                name='Средний по вузу',
                line=dict(color='rgba(128, 128, 128, 0.7)', width=1),
                fillcolor='rgba(128, 128, 128, 0.2)'
            ))

            # Слой студента
            fig_radar.add_trace(go.Scatterpolar(
                r=[val_gpa, val_discipline, val_experience, val_stability, val_gpa],
                theta=categories + [categories[0]],
                fill='toself',
                name='Данный студент',
                line=dict(color='#4169E1', width=3),
                fillcolor='rgba(65, 105, 225, 0.3)',
                marker=dict(size=8)
            ))

            fig_radar.update_layout(
                polar=dict(
                    # ПРОЗРАЧНЫЙ ФОН САМОГО РАДАРА
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        gridcolor='rgba(128, 128, 128, 0.5)',
                        linecolor='rgba(128, 128, 128, 0.5)',
                        tickfont=dict(size=10, color='gray')
                    ),
                    angularaxis=dict(
                        gridcolor='rgba(128, 128, 128, 0.5)',
                        linecolor='rgba(128, 128, 128, 0.5)'
                    )
                ),

                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                height=550,
                margin=dict(l=80, r=80, t=20, b=20)
            )

            st.plotly_chart(fig_radar, use_container_width=True, theme="streamlit")

        except Exception as e:
            st.info("Недостаточно данных для построения профиля компетенций.")


# ВКЛАДКА 2: МАССОВЫЙ РАСЧЕТ С ПРОГРЕСС-БАРОМ
with tab2:
    st.markdown("### Пакетная обработка и выгрузка результатов")
    st.write(f"Готово к анализу: **{len(df)}** записей.")

    if st.button("Запустить массовый расчет", type="primary"):

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.text("Подготовка признаков...")
            X_all = prepare_features(df, saved_columns=saved_columns)

            status_text.text("Модель формирует прогнозы...")
            preds = model.predict(X_all)

            result_df = df.copy()
            result_df["ПРОГНОЗ"] = preds

            status_text.text("Расчет ключевых факторов риска (SHAP)...")

            explainer = shap.TreeExplainer(model)
            feature_names = X_all.columns.tolist()

            result_df["ГЛАВНЫЙ_ФАКТОР"] = ""
            result_df["ВЛИЯНИЕ_ФАКТОРА"] = ""

            total_rows = len(X_all)
            chunk_size = max(1, math.ceil(total_rows / 20))

            for start_idx in range(0, total_rows, chunk_size):
                end_idx = min(start_idx + chunk_size, total_rows)
                X_chunk = X_all.iloc[start_idx:end_idx]

                shap_chunk = explainer.shap_values(X_chunk)

                for i_chunk, i_global in enumerate(range(start_idx, end_idx)):
                    if model_type == 0:
                        sample = shap_chunk[1][i_chunk] if isinstance(shap_chunk, list) else shap_chunk[i_chunk]
                    else:
                        cls = preds[i_global]
                        cls_idx = list(model.classes_).index(cls)
                        sample = shap_chunk[cls_idx][i_chunk] if isinstance(shap_chunk, list) else shap_chunk[
                            i_chunk, :, cls_idx]

                    # Топ-1 фактор
                    top_idx = np.argsort(np.abs(sample))[::-1][0]
                    impact = sample[top_idx]

                    result_df.loc[i_global, "ГЛАВНЫЙ_ФАКТОР"] = feature_names[top_idx]
                    result_df.loc[i_global, "ВЛИЯНИЕ_ФАКТОРА"] = "Тянет ВНИЗ 🔻" if impact < 0 else "Тянет ВВЕРХ 🔺"

                # Обновляем
                progress = end_idx / total_rows
                progress_bar.progress(progress)
                status_text.text(f"Обработано {end_idx} из {total_rows} студентов...")

            status_text.text("✅ Расчет успешно завершен!")
            progress_bar.empty()

            st.subheader("Предпросмотр результатов")
            st.dataframe(result_df.head(10), use_container_width=True)

            csv_buffer = result_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
            st.download_button(
                label="Скачать полный отчет (.csv)",
                data=csv_buffer,
                file_name="Mass_Prediction_Results.csv",
                mime="text/csv",
            )

        except Exception as e:
            status_text.empty()
            st.error(f"Произошла ошибка при массовом расчете: {e}")

# ВКЛАДКА 3: РУЧНОЙ ВВОД (СИМУЛЯТОР)
with tab3:
    st.markdown("### Ввод данных вручную")
    st.info(
        "Используйте эту форму, если студента еще нет в базе или вы хотите проверить гипотезу (например, 'что будет, если GPA станет 4.5').")

    # with st.form("manual_input_form"):
    #     col_m1, col_m2 = st.columns(2)
    #
    #     input_data = {}
    #
    #     for i, col in enumerate(saved_columns):
    #         target_col = col_m1 if i % 2 == 0 else col_m2
    #
    #         with target_col:
    #             if col in ['ПОЛ', 'ПРИЗНАК_ОБЩЕЖИТИЯ', 'УСЛОВИЕ_ЗАЧИСЛЕНИЯ', 'ГРАЖДАНСТВО', 'ДИСЦИПЛИНА']:
    #                 try:
    #                     from model_utils import MODEL_DIR
    #                     import joblib
    #
    #                     le = joblib.load(
    #                         f"models/{'binary' if model_type == 0 else 'quality'}_columns.pkl")  # тут логика может отличаться
    #                     options = ["Вариант 1", "Вариант 2"]
    #                 except:
    #                     options = ["Бюджет", "Конкурс", "Целевая квота", "Платное"]
    #
    #                 input_data[col] = st.selectbox(f"Выберите {col}",
    #                                                options=["Бюджет", "Конкурс", "Платное"] if "УСЛОВИЕ" in col else [
    #                                                    "М", "Ж"] if col == "ПОЛ" else ["Да", "Нет"])
    #
    #             elif "Средняя" in col or "GPA" in col or "Макс" in col or "Мин" in col:
    #                 input_data[col] = st.slider(f"{col}", 2.0, 5.0, 3.5, 0.1)
    #
    #             elif "СЕМЕСТР" in col or "Семестр" in col:
    #                 input_data[col] = st.number_input(f"{col}", 1, 8, 1)
    #
    #             elif "Несдано" in col or "Число" in col or "Всего" in col:
    #                 input_data[col] = st.number_input(f"{col}", 0, 50, 0)
    #
    #             else:
    #                 input_data[col] = st.text_input(f"Введите {col}", "0")
    #
    #     submit_manual = st.form_submit_button("Рассчитать прогноз для введенных данных", type="primary",
    #                                           use_container_width=True)
    #
    # if submit_manual:
    #     try:
    #         manual_df = pd.DataFrame([input_data])
    #
    #         X_manual = prepare_features(manual_df, saved_columns=saved_columns)
    #
    #         pred_val = model.predict(X_manual)[0]
    #
    #         st.divider()
    #
    #         res_col1, res_col2 = st.columns([1, 2])
    #
    #         with res_col1:
    #             st.markdown("### Результат:")
    #             if model_type == 0:
    #                 status = "СДАСТ 🟢" if pred_val == 1 else "НЕ СДАСТ 🔴"
    #                 st.title(status)
    #             else:
    #                 st.title(f"Оценка: {pred_val}")
    #
    #             if hasattr(model, "predict_proba"):
    #                 proba = model.predict_proba(X_manual)[0]
    #                 conf = proba[int(pred_val)] if model_type == 1 else (proba[1] if pred_val == 1 else proba[0])
    #                 st.metric("Уверенность модели", f"{conf * 100:.1f}%")
    #
    #         with res_col2:
    #             st.markdown("### Детализация погноза")
    #
    #             explainer = shap.TreeExplainer(model)
    #             shap_values_manual = explainer.shap_values(X_manual)
    #
    #             if model_type == 0:
    #                 sv = shap_values_manual[1][0] if isinstance(shap_values_manual, list) else shap_values_manual[0]
    #                 bv = explainer.expected_value[1] if isinstance(explainer.expected_value,
    #                                                                (list, np.ndarray)) else explainer.expected_value
    #             else:
    #                 cls_idx = list(model.classes_).index(pred_val)
    #                 sv = shap_values_manual[cls_idx][0] if isinstance(shap_values_manual, list) else shap_values_manual[
    #                     0, :, cls_idx]
    #                 bv = explainer.expected_value[cls_idx] if isinstance(explainer.expected_value, (list,
    #                                                                                                 np.ndarray)) else explainer.expected_value
    #
    #             fig_m, ax_m = plt.subplots(figsize=(10, 4))
    #             exp_m = shap.Explanation(
    #                 values=sv, base_values=bv, data=X_manual.iloc[0].values,
    #                 feature_names=X_manual.columns.tolist()
    #             )
    #             shap.plots.waterfall(exp_m, show=False)
    #             plt.tight_layout()
    #             st.pyplot(fig_m)
    #             plt.close(fig_m)
    #
    #     except Exception as e:
    #         st.error(f"Ошибка при расчете: {e}")
