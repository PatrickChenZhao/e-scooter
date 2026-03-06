import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import cv2          
import tempfile
from openai import OpenAI
import json
import time

# --- 1. 页面基本设置 ---
st.set_page_config(page_title="Vision Dashboard", layout="wide", initial_sidebar_state="collapsed")

# --- 2. 前端 CSS (融合你的设计灵感与终极修复) ---
st.markdown("""
    <style>
    /* 全局炫酷渐变背景 */
    .stApp {
        background: linear-gradient(120deg, #D02090 0%, #483D8B 50%, #00FF00 100%);
        color: #E0E4E8;
    }

    h1, h3, h4 { color: #FFFFFF; padding-bottom: 10px; }

    /* KPI 卡片 */
    div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px; padding: 12px; border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* 修复下拉菜单 */
    .stSelectbox > div > div > div { background-color: #1a2536 !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; color: #E0E4E8 !important; }
    .stSlider > div > div > div > div > div { background-color: rgba(255, 255, 255, 0.05) !important; }

    /* 修复文件上传区：完美对齐暗色调 */
    [data-testid="stFileUploader"] > section {
        background-color: #1a2536 !important; 
        border: 1px dashed rgba(255, 255, 255, 0.3) !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploader"] > section span, [data-testid="stFileUploader"] > section small { color: #E0E4E8 !important; }
    [data-testid="stFileUploader"] > section button { background-color: rgba(255, 255, 255, 0.05) !important; color: #FFFFFF !important; border: 1px solid rgba(255, 255, 255, 0.2) !important; }

    /* 修复按钮：深色磨砂+悬停发光特效 */
    .stButton > button {
        background-color: rgba(0, 0, 0, 0.5) !important; color: #FFFFFF !important; 
        border: 1px solid rgba(255, 255, 255, 0.3) !important; border-radius: 8px !important; transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        background-color: #D02090 !important; border: 1px solid #FFFFFF !important; box-shadow: 0 0 10px rgba(208, 32, 144, 0.5) !important;
    }

    /* 【核心绝杀】：实现你的大背景板设想！将整个 Tab 内容区变成一个深色磨砂玻璃容器 */
    div[role="tabpanel"] {
        background-color: rgba(0, 0, 0, 0.15) !important;
        padding: 30px !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }

    /* 图片占位符 */
    .placeholder-box {
        border: 2px dashed rgba(255, 255, 255, 0.2); border-radius: 8px; height: 300px;
        display: flex; align-items: center; justify-content: center;
        color: rgba(255, 255, 255, 0.3); background-color: rgba(255, 255, 255, 0.02);
    }

    /* ==========================================
       UI 终极优化：字体大小、颜色、页面留白布局
       ========================================== */

    /* 【修改 1】放大 Tab 标签页标题字体 */
    button[data-baseweb="tab"] > div {
        font-size: 20px !important;
        font-weight: bold !important;
    }

    /* 【终极修改 2】暴力覆写：强制所有组件标签（Upload Image, Confidence Threshold）变为纯白 */
    .stFileUploader label p, .stFileUploader label div, 
    .stSlider label p, .stSlider label div,
    label[data-testid="stWidgetLabel"] p {
        color: #FFFFFF !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.8) !important; /* 增加高级黑色投影，彻底消除看不清的问题 */
    }
    
    /* 【终极修改 2】暴力覆写：强制滑动条下方的 0.00, 1.00 刻度变为纯白 */
    div[data-testid="stTickBar"] div,
    div[data-testid="stSliderTickBarMin"], 
    div[data-testid="stSliderTickBarMax"] {
        color: #FFFFFF !important;
        font-weight: bold !important;
        font-size: 14px !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.8) !important;
    }

    /* 【修改 4】全局居中，固定宽度，左右强制留白 10% */
    .block-container {
        max-width: 80vw !important; /* 强制内容最大宽度为屏幕的 80% */
        margin: 0 auto !important;  /* 自动居中，左右自然各剩下 10% 的完美留白 */
    }
            
    
    </style>
""", unsafe_allow_html=True)

morandi_colors = ['#8A9A9A', '#B0A8B9', '#C0B2AB', '#7F8D8B', '#A39391', '#6B7A7F']

# 主标题与副标题全局居中，并优化上下间距
st.markdown("<h1 style='text-align: center; color: #FFFFFF;'>E-Scooter Vision Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #E0E4E8; font-size: 1.2rem; margin-top: -10px;'>Simple Scene Understanding & Video Tracking Engine</p>", unsafe_allow_html=True)
st.markdown("---")

# --- 3. 核心双模型加载 ---
@st.cache_resource # 恢复极其安全的缓存，因为我们现在分离了模型任务，不会再冲突了
def load_models():
    base_m = YOLO('yolov8n-seg.pt') 
    custom_m = YOLO('nz_scooter_model.pt')
    return base_m, custom_m

model_base, model_custom = load_models()
urban_classes = [0, 1, 2, 3, 5, 7, 9, 11]

# ==========================================
# 4. 标签页划分 (自带全局底板特效)
# ==========================================
tab_image, tab_video = st.tabs(["Scene Understanding from Image(Dual-Core)", "Video Tracking（e-scooter）"])

# ------------------------------------------
# 标签页 1：静态图像推断
# ------------------------------------------
with tab_image:
    st.markdown("### Control Panel (Image Analysis)")
    img_ctrl_col1, img_ctrl_col2, img_ctrl_col3 = st.columns(3)
    
    with img_ctrl_col1:
        uploaded_img = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], key="img_uploader")
    with img_ctrl_col2:
        conf_img = st.slider(
            " Confidence Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.25, 
            step=0.05, 
            help="Minimum confidence score for object detection. Higher values reduce false positives but may miss valid objects.",
            key="img_conf"
        )
    with img_ctrl_col3:
        st.markdown("**Model Selection**")
        model_choice_img = st.selectbox("Choose Neural Network", ["Dual-Core Engine (YOLOv8n + E-scooter)"], index=0, help="Simultaneously running standard objects and custom e-scooters.", key="img_model_select")

    st.markdown("<br>", unsafe_allow_html=True) # 加一点间距，让呼吸感更好
    main_col1, main_col2 = st.columns([3, 1], gap="large")
    
    with main_col1:
        img_view1, img_view2 = st.columns(2)
        if uploaded_img is None:
            with img_view1: st.markdown('<div class="placeholder-box">Original Image Placeholder</div>', unsafe_allow_html=True)
            with img_view2: st.markdown('<div class="placeholder-box">Analyzed Output Placeholder</div>', unsafe_allow_html=True)
        else:
            image = Image.open(uploaded_img).convert('RGB')
            img_array = np.array(image)
            
            with st.spinner("Dual-Neural Engine Processing..."):
                results_base = model_base.predict(source=img_array, conf=conf_img, classes=urban_classes, verbose=False)
                results_custom = model_custom.predict(source=img_array, conf=conf_img, verbose=False)
                
                img_with_base = results_base[0].plot()
                res_image_bgr = results_custom[0].plot(img=img_with_base)
                res_image = res_image_bgr[:, :, ::-1] 
            
            with img_view1:
                st.markdown("#### Original Image")
                st.image(image, use_container_width=True)
            with img_view2:
                st.markdown("#### Analyzed Output (Fused)")
                st.image(res_image, use_container_width=True)
        
        # 场景推断 (AI DeepSeek)
        if uploaded_img is not None:
            st.markdown("---")
            st.markdown("#### AI Scene Intelligence")
            
            cls_base = results_base[0].boxes.cls.cpu().numpy()
            cls_custom = results_custom[0].boxes.cls.cpu().numpy()
            class_names_base = [model_base.names[int(c)] for c in cls_base]
            class_names_custom = [model_custom.names[int(c)] for c in cls_custom]
            
            final_detections = {
                "Standard Transportation": class_names_base,
                "Custom E-Scooter (Your Model)": class_names_custom
            }
            
            try:
                # 读取在 Streamlit Secrets 里填写的密码
                api_key = st.secrets["DEEPSEEK_API_KEY"]
                client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            except Exception as e:
                st.error("🔑 尚未配置 AI API Key，请在 Streamlit 设置的 Secrets 中添加 DEEPSEEK_API_KEY。")
                st.stop()

            # 提示词输出结构
            system_prompt = """You are an expert urban transportation analyst in Auckland, New Zealand. 
            Based on a structured list of objects detected from a street image, generate a concise and professional scene description.
            CRITICAL REQUIREMENT: You MUST output your analysis in BOTH English and Chinese. 
            Please format your response exactly like this:
            
            English Analysis:
            [Your English description here]
            

            中文分析:
            [Your Chinese description here]
            
            Keep it professional, engaging, and focus on the dynamics of urban micro-mobility and standard traffic."""
            
            user_prompt = f"""Detections List: {final_detections}"""
            
            with st.spinner("DeepSeek AI 正在分析城市交通场景..."):
                try:
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.7 
                    )
                    ai_description = response.choices[0].message.content
                    
                    # 【UI 升级】使用 HTML/CSS 强制渲染纯白色字体，并加上微透明的毛玻璃背景框
                    st.markdown(f"""
                        <div style='color: #FFFFFF; background-color: rgba(255, 255, 255, 0.08); 
                                    padding: 20px; border-radius: 12px; border: 1px solid rgba(255, 255, 255, 0.2); 
                                    line-height: 1.6; font-size: 16px; text-shadow: 1px 1px 3px rgba(0,0,0,0.9);'>
                            {ai_description}
                        </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ AI failed 分析失败: {e}")

    with main_col2:
            st.markdown("#### Insights & Analytics")
            if uploaded_img is not None:
                cls_base = results_base[0].boxes.cls.cpu().numpy()
                cls_custom = results_custom[0].boxes.cls.cpu().numpy()
                class_names_base = [model_base.names[int(c)] for c in cls_base]
                class_names_custom = [model_custom.names[int(c)] for c in cls_custom]
                all_class_names = class_names_base + class_names_custom
                
                if len(all_class_names) > 0:
                    unique_classes, counts = np.unique(all_class_names, return_counts=True)
                    kpi1, kpi2 = st.columns(2)
                    kpi1.metric("Total Objects", len(all_class_names))
                    kpi2.metric("Categories", len(unique_classes))
                    
                    df_counts = pd.DataFrame({'Object': all_class_names}).value_counts().rename_axis('Object').reset_index(name='Count')
                    fig_bar = px.bar(df_counts, x='Count', y='Object', orientation='h', color='Object', color_discrete_sequence=morandi_colors)
                    
                    # 【修改 1：柱状图文字放大并变成黑色】
                    fig_bar.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)', 
                        paper_bgcolor='rgba(0,0,0,0)', 
                        font=dict(color='#000000', size=16, family="Arial, sans-serif"), 
                        xaxis=dict(title_font=dict(size=18, color="black"), tickfont=dict(size=14, color="black")),
                        yaxis=dict(title_font=dict(size=18, color="black"), tickfont=dict(size=14, color="black")),
                        showlegend=False, 
                        height=280, 
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    all_objs_pie, all_areas_pie = [], []
                    if results_base[0].masks is not None:
                        m_b = results_base[0].masks.data.cpu().numpy()
                        all_areas_pie.extend(m_b.sum(axis=(1, 2)))
                        all_objs_pie.extend(class_names_base)
                    if results_custom[0].masks is not None:
                        m_c = results_custom[0].masks.data.cpu().numpy()
                        all_areas_pie.extend(m_c.sum(axis=(1, 2)))
                        all_objs_pie.extend(class_names_custom)
                        
                    if len(all_objs_pie) > 0:
                        df_area = pd.DataFrame({'Object': all_objs_pie, 'Pixel Area': all_areas_pie})
                        df_area_sum = df_area.groupby('Object')['Pixel Area'].sum().reset_index()
                        fig_pie = px.pie(df_area_sum, values='Pixel Area', names='Object', hole=0.5, color_discrete_sequence=morandi_colors)
                        
                        # 【修改 2：饼图图例和标签文字放大并变成黑色】
                        fig_pie.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)', 
                            paper_bgcolor='rgba(0,0,0,0)', 
                            font=dict(color='#000000', size=15, family="Arial, sans-serif"), 
                            height=300, 
                            margin=dict(l=0, r=0, t=30, b=0), 
                            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
                        )
                        # 让饼图内部的文字不仅显示百分比，还显示名称，且变为 16 号黑色
                        fig_pie.update_traces(
                            textposition='inside', 
                            textinfo='percent+label', 
                            textfont=dict(color='#000000', size=16)
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.caption("No mask areas to render pie chart.")
                else:
                    st.warning("No objects detected.")
            else:
                st.warning("Please upload an image in the Control Panel.")


# ------------------------------------------
# 标签页 2：动态视频跟踪引擎
# ------------------------------------------
with tab_video:
    st.markdown("###  Control Panel (Video Tracking)")
    vid_ctrl_col1, vid_ctrl_col2, vid_ctrl_col3 = st.columns(3)
    
    with vid_ctrl_col1:
        uploaded_vid = st.file_uploader("Upload Street Video", type=['mp4', 'mov', 'avi'], key="vid_uploader")
    with vid_ctrl_col2:
        conf_vid = st.slider(
            "Confidence Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.25, 
            step=0.05, 
            help="Minimum confidence score for object detection. Higher values reduce false positives but may miss valid objects.",
            key="vid_conf"
        )
    with vid_ctrl_col3:
        st.markdown("**Tracking Engine**")
        model_choice_vid = st.selectbox("Choose Tracking System", ["BoT-SORT Engine (E-scooter+YOLOv8)"], index=0, help="Powered by your custom model and BoT-SORT algorithm for motion compensation.", key="vid_model_select")

    st.markdown("<br>", unsafe_allow_html=True)
    
    if uploaded_vid is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_vid.read())
        vf = cv2.VideoCapture(tfile.name)
        total_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(vf.get(cv2.CAP_PROP_FPS))
        
        st.info(f"Video loaded：total {total_frames} frames | Frame rate: {fps} FPS")
        
        vid_col1, vid_col2 = st.columns([3, 1], gap="large")
        
        with vid_col2:
            status_title_placeholder = st.empty()
            progress_bar_placeholder = st.empty()
            status_text = st.empty()
            
        with vid_col1:
            spacer_left, center_video_col, spacer_right = st.columns([1, 6, 1])
            with center_video_col:
                video_placeholder = st.empty()
        
        action_button_container = st.empty()
        
        if action_button_container.button("启动跟踪引擎 | Start Tracking)", use_container_width=True):
            action_button_container.empty()
            status_title_placeholder.markdown("#### ⚙️Tracking Status")
            progress_bar = progress_bar_placeholder.progress(0)
            
            

            frame_count = 0
            while vf.isOpened():
                ret, frame = vf.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # ==========================================
                # 🚀 【安全门卫】：如果不是偶数帧，直接略过！
                # ==========================================
                if frame_count % 2 != 0: 
                    continue
                
                # ------ 以下代码，每两帧才被允许执行一次 ------
                
                # 1. 更新进度条 (加了 min 防护，绝对不会报错)
                progress_bar.progress(min(frame_count / total_frames, 1.0))
                status_text.write(f"Processed: **{frame_count}** / {total_frames} frames")
                
                # 2. AI 追踪与画图 (锁定 480 尺寸)
                results = model_custom.track(frame, persist=True, conf=conf_vid, tracker="botsort.yaml", imgsz=480, verbose=False)
                annotated_frame = results[0].plot()
                
                # 3. 画面体积压缩
                max_height = 480
                h, w = annotated_frame.shape[:2]
                if h > max_height:
                    scale = max_height / h
                    annotated_frame = cv2.resize(annotated_frame, (int(w * scale), max_height))
                    
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # 4. 把画面塞给前端显示
                video_placeholder.image(rgb_frame, channels="RGB", use_container_width=False)
                
                # ==========================================
                # 🚀 【终极杀手锏：强制喘息机制】
                # 强迫 AI 闭嘴 0.05 秒！把 CPU 让给 Streamlit，
                # 让它有时间把刚才那张图片通过网络发给你的浏览器！
                # ==========================================
                time.sleep(0.05)
                
            # 循环跑完后的收尾动作
            progress_bar.progress(1.0)
            status_text.write(f"Processed: **{total_frames}** / {total_frames} frames")
            vf.release()
            st.success("Processing Completed")
            
            if action_button_container.button("重新分析 / Reset Dashboard", use_container_width=True):
                st.rerun() 
    else:
        st.warning("Please upload a video in the Control Panel.")