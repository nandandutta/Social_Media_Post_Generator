import streamlit as st
from prompts import get_text_prompt, get_image_prompt
from model import generate_image_from_text,generate_text_from_prompt,add_watermark

def main():
    st.title("Social Media Post Generator")
    
    st.markdown(''' :red[Social] :orange[Media] :green[Post] :blue[Generator].''')
    
    tab1, tab2 = st.tabs(["Post Generation", "Video Generation(working on....)"])

    with tab1:
        st.markdown("""
    **Sample Prompts:**
    1. **Durgapuja Pandal:** A traditional Bengali Durgapuja pandal with intricate decorations, flowers, and a stunning idol of Goddess Durga.
    2. **Goddess Durga:** Goddess Durga defeating the buffalo-demon Mahishasura, with a fiery explosion in the background.
    3. **Durgapuja Festival:** A vibrant Durgapuja festival scene with people dressed in traditional attire, enjoying food, music, and dance.
""")


        
        occasion = st.text_input("Enter the Prompts", key="occasion_tab1")
        st.markdown('''
    :red[A] :orange[traditional] :green[Bengali] :blue[Durgapuja] :violet[pandal]
    :gray[with] :rainbow[intricate] :blue[decorations], :green[flowers], :orange[and]
    :red[a] :blue[stunning] :green[idol] :violet[of] :orange[Goddess] :red[Durga].
''')
        company_type = st.selectbox("Select Company Type", ["Tech Startup", "Retail", "Healthcare", "Finance", "Other"], key="company_type_tab1")
        post_type = st.selectbox("Select Post Type", [
            "General Announcement", "Product Launch", "Event Promotion",
            "Special Promotion or Discount", "Customer Testimonials or Case Studies",
            "Company Milestone Celebration", "Seasonal or Holiday Greetings",
            "Recruitment or Job Posting", "Brand Awareness or Introduction",
            "Corporate Social Responsibility (CSR) Initiative"
        ], key="post_type_tab1")
        visual_style = st.selectbox("Select Visual Style", ["Modern and sleek", "Vintage", "Minimalist", "Bold and colorful"], key="visual_style_tab1")
        color_scheme = st.selectbox("Select Color Scheme", ["Tech-themed colors", "Warm colors", "Cool colors", "Custom"], key="color_scheme_tab1")
        mood = st.selectbox("Select Mood/Emotion", ["Excitement", "Professional", "Fun", "Elegant"], key="mood_tab1")
        watermark_text = st.text_input("Enter Company Name", value=" Watermark", key="watermark_text_tab1")

        if st.button("Generate Post", key="generate_post_tab1"):
            with st.spinner("Generating content..."):
                # Create the prompts
                text_prompt = get_text_prompt(occasion, company_type, post_type, visual_style, color_scheme, mood,watermark_text)
                image_prompt = get_image_prompt(occasion, company_type, post_type, visual_style, color_scheme, mood,watermark_text)

                # Generate text
                text_content = generate_text_from_prompt(text_prompt, "")
                if text_content:
                    st.text(text_content)

                # Generate image
                generated_image = generate_image_from_text(image_prompt)
                if generated_image:
                    # Add watermark
                    watermarked_image = add_watermark(generated_image, watermark_text=watermark_text)

                    # Display image
                    st.subheader("Generated Image with Watermark")
                    st.image(watermarked_image, caption='Generated Image with Watermark')

    with tab2:
        occasion = st.text_input("Enter the Occasion (e.g., 'Product Launch')", key="occasion_tab2")
        company_type = st.selectbox("Select Company Type", ["Tech Startup", "Retail", "Healthcare", "Finance", "Other"], key="company_type_tab2")
        post_type = st.selectbox("Select Post Type", [
            "General Announcement", "Product Launch", "Event Promotion",
            "Special Promotion or Discount", "Customer Testimonials or Case Studies",
            "Company Milestone Celebration", "Seasonal or Holiday Greetings",
            "Recruitment or Job Posting", "Brand Awareness or Introduction",
            "Corporate Social Responsibility (CSR) Initiative"
        ], key="post_type_tab2")
        visual_style = st.selectbox("Select Visual Style", ["Modern and sleek", "Vintage", "Minimalist", "Bold and colorful"], key="visual_style_tab2")
        color_scheme = st.selectbox("Select Color Scheme", ["Tech-themed colors", "Warm colors", "Cool colors", "Custom"], key="color_scheme_tab2")
        mood = st.selectbox("Select Mood/Emotion", ["Excitement", "Professional", "Fun", "Elegant"], key="mood_tab2")
        watermark_text = st.text_input("Enter Watermark Text", value="Sample Watermark", key="watermark_text_tab2")

        if st.button("Generate Post", key="generate_post_tab2"):
            with st.spinner("Generating content..."):
                # Create the prompts
                text_prompt = get_text_prompt(occasion, company_type, post_type, visual_style, color_scheme, mood)
                image_prompt = get_image_prompt( occasion,company_type, post_type, visual_style, color_scheme, mood)

                # Generate text
                text_content = generate_text_from_prompt(text_prompt, "")
                if text_content:
                    st.text(text_content)
                    st.session_state.generated_text = text_content

                # Generate image
                generated_image = generate_image_from_text(text_content,image_prompt)
                if generated_image:
                    # Add watermark
                    watermarked_image = add_watermark(generated_image, watermark_text=watermark_text)

                    # Display image
                    st.subheader("Generated Image with Watermark")
                    st.image(watermarked_image, caption='Generated Image with Watermark')

if __name__ == "__main__":
    main()
