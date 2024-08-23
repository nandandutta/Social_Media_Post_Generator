# # prompts.py

# def get_text_prompt(occasion, company_type, post_type, visual_style, color_scheme, watermark_text,mood):
#     prompts = {
#         "General Announcement": (
#             f"Write an engaging announcement for a {company_type} company about {occasion} and company name {watermark_text}. "
#             f"The content should be professional, concise, and include key details that will capture the audience's attention. "
#             f"Make sure the tone matches the brand's identity."
#         ),
#         "Product Launch": (
#             f"Craft a compelling product launch post for a {company_type} company introducing {occasion}and company name {watermark_text}. "
#             f"The text should highlight the product's unique features and benefits, explaining how it addresses customer needs. "
#             f"The tone should be {mood}, aiming to create excitement and anticipation among the audience."
#         ),
#         "Event Promotion": (
#             f"Generate promotional content for an upcoming {occasion} hosted by a {company_type} company and company name {watermark_text}. "
#             f"The post should include key details like date, time, location, and any special guests or activities. "
#             f"The tone should be {mood}, aimed at encouraging maximum attendance."
#         ),
#         "Special Promotion or Discount": (
#             f"Write a persuasive social media post for a {company_type} company announcing a {occasion} promotion and company name {watermark_text}. "
#             f"The content should clearly state the discount or special offer, the duration, and how customers can take advantage of it. "
#             f"The tone should be {mood} and should appeal to the target audience's desire for value."
#         ),
#         "Customer Testimonials or Case Studies": (
#             f"Create a testimonial post for a {company_type} company, featuring a satisfied customer's feedback and company name {watermark_text}. "
#             f"The content should highlight the customer's experience with {occasion}, emphasizing the positive impact it had. "
#             f"The tone should be {mood} to build trust and credibility."
#         ),
#         "Company Milestone Celebration": (
#             f"Draft a celebratory post for a {company_type} company marking {occasion} and company name {watermark_text}. "
#             f"The content should express gratitude to customers and employees, and highlight the significance of the achievement. "
#             f"The tone should be {mood}, creating a sense of pride and accomplishment."
#         ),
#         "Seasonal or Holiday Greetings": (
#             f"Compose a warm holiday greeting post for a {company_type} company and company name {watermark_text}. "
#             f"The message should wish the audience a happy {occasion}, while subtly aligning with the brand's identity. "
#             f"The tone should be {mood} and convey a sense of goodwill and festivity."
#         ),
#         "Recruitment or Job Posting": (
#             f"Create an inviting job posting for a {company_type} company looking to hire for the position of {occasion} and company name {watermark_text}. "
#             f"The content should outline key responsibilities, qualifications, and what makes the company a great place to work. "
#             f"The tone should be {mood} to attract the right candidates."
#         ),
#         "Brand Awareness or Introduction": (
#             f"Write a brand introduction post for a {company_type} company and company name {watermark_text}. "
#             f"The content should briefly explain the company's mission, values, and what sets it apart from competitors. "
#             f"The tone should be {mood}, aiming to build connection and interest among potential customers."
#         ),
#         "Corporate Social Responsibility (CSR) Initiative": (
#             f"Craft a post about a CSR initiative undertaken by a {company_type} company and company name {watermark_text}. "
#             f"The content should describe the initiative, its impact, and why it matters to the community and the company. "
#             f"The tone should be {mood} to resonate with socially conscious audiences."
#         )
#     }
#     return prompts[post_type]

# def get_image_prompt(occasion, company_type, post_type, visual_style, color_scheme,watermark_text, mood):
#     prompts = {
#         "General Announcement": (
#             f"Create a visually striking image for a social media post announcing {occasion} for a {company_type} company and company name {watermark_text}. "
#             f"The style should be {visual_style} with a {color_scheme} color scheme, conveying a mood of {mood}. "
#             f"Include elements that highlight the announcement's significance."
#         ),
#         "Product Launch": (
#             f"Design a dynamic and engaging image for the launch of {occasion} by a {company_type} company and company name {watermark_text}. "
#             f"The visual should reflect a {visual_style} with {color_scheme} colors and convey a sense of {mood}. "
#             f"Include product imagery and branding elements that enhance the launch message."
#         ),
#         "Event Promotion": (
#             f"Create an attractive promotional image for a {occasion} organized by a {company_type} company and company name {watermark_text}. "
#             f"The image should be in a {visual_style} with a {color_scheme} palette, and should evoke feelings of {mood}. "
#             f"Incorporate event-specific elements like logos, themes, or relevant visuals."
#         ),
#         "Special Promotion or Discount": (
#             f"Design an eye-catching image to promote a {occasion} discount for a {company_type} company and company name {watermark_text}. "
#             f"The image should feature a {visual_style} with a {color_scheme} that grabs attention, conveying a sense of {mood}. "
#             f"Include promotional elements like discount percentages or offer details."
#         ),
#         "Customer Testimonials or Case Studies": (
#             f"Generate an image to accompany a customer testimonial for a {company_type} company and company name {watermark_text}. "
#             f"The visual should be in a {visual_style} with {color_scheme} colors, reflecting a {mood}. "
#             f"Include elements that reinforce the testimonial, like customer photos or before-and-after visuals."
#         ),
#         "Company Milestone Celebration": (
#             f"Create a celebratory image to mark {occasion} for a {company_type} company and company name {watermark_text}. "
#             f"The image should have a {visual_style} with a {color_scheme} color palette, and convey {mood}. "
#             f"Incorporate milestone-specific graphics like numbers, trophies, or company logos."
#         ),
#         "Seasonal or Holiday Greetings": (
#             f"Design a festive image for a {occasion} greeting from a {company_type} companyand company name {watermark_text}. "
#             f"The image should reflect a {visual_style} with a {color_scheme} palette, evoking {mood}. "
#             f"Include holiday-related elements like snowflakes, lights, or traditional symbols."
#         ),
#         "Recruitment or Job Posting": (
#             f"Generate a professional and appealing image to accompany a job posting for the role of {occasion} at a {company_type} company and company name {watermark_text}. "
#             f"The image should have a {visual_style} with {color_scheme} colors, and convey a {mood} that aligns with the company's culture."
#         ),
#         "Brand Awareness or Introduction": (
#             f"Create an engaging image for a brand awareness post about a {company_type} company and company name {watermark_text}. "
#             f"The visual should use a {visual_style} with a {color_scheme} palette, and convey {mood}. "
#             f"Include branding elements like the logo, tagline, and key visuals that represent the company."
#         ),
#         "Corporate Social Responsibility (CSR) Initiative": (
#             f"Design a supportive and inspiring image for a post about a CSR initiative by a {company_type} company and company name {watermark_text}. "
#             f"The image should reflect a {visual_style} with {color_scheme} colors, and convey {mood}. "
#             f"Include elements like project photos, community engagement, or impact statistics."
#         )
#     }
#     return prompts[post_type]




# prompts.py

def get_text_prompt(occasion, company_type, post_type, visual_style, color_scheme, mood, watermark_text):
    prompts = {
        "General Announcement": (
            f"Write an engaging announcement for a {company_type} company about {occasion} and company name {watermark_text}. "
            f"The content should be professional, concise, and include key details that will capture the audience's attention. "
            f"Make sure the tone matches the brand's identity."
        ),
        "Product Launch": (
            f"Craft a compelling product launch post for a {company_type} company introducing {occasion} and company name {watermark_text}. "
            f"The text should highlight the product's unique features and benefits, explaining how it addresses customer needs. "
            f"The tone should be {mood}, aiming to create excitement and anticipation among the audience."
        ),
        "Event Promotion": (
            f"Generate promotional content for an upcoming {occasion} hosted by a {company_type} company and company name {watermark_text}. "
            f"The post should include key details like date, time, location, and any special guests or activities. "
            f"The tone should be {mood}, aimed at encouraging maximum attendance."
        ),
        "Special Promotion or Discount": (
            f"Write a persuasive social media post for a {company_type} company announcing a {occasion} promotion and company name {watermark_text}. "
            f"The content should clearly state the discount or special offer, the duration, and how customers can take advantage of it. "
            f"The tone should be {mood} and should appeal to the target audience's desire for value."
        ),
        "Customer Testimonials or Case Studies": (
            f"Create a testimonial post for a {company_type} company, featuring a satisfied customer's feedback and company name {watermark_text}. "
            f"The content should highlight the customer's experience with {occasion}, emphasizing the positive impact it had. "
            f"The tone should be {mood} to build trust and credibility."
        ),
        "Company Milestone Celebration": (
            f"Draft a celebratory post for a {company_type} company marking {occasion} and company name {watermark_text}. "
            f"The content should express gratitude to customers and employees, and highlight the significance of the achievement. "
            f"The tone should be {mood}, creating a sense of pride and accomplishment."
        ),
        "Seasonal or Holiday Greetings": (
            f"Compose a warm holiday greeting post for a {company_type} company and company name {watermark_text}. "
            f"The message should wish the audience a happy {occasion}, while subtly aligning with the brand's identity. "
            f"The tone should be {mood} and convey a sense of goodwill and festivity."
        ),
        "Recruitment or Job Posting": (
            f"Create an inviting job posting for a {company_type} company looking to hire for the position of {occasion} and company name {watermark_text}. "
            f"The content should outline key responsibilities, qualifications, and what makes the company a great place to work. "
            f"The tone should be {mood} to attract the right candidates."
        ),
        "Brand Awareness or Introduction": (
            f"Write a brand introduction post for a {company_type} company and company name {watermark_text}. "
            f"The content should briefly explain the company's mission, values, and what sets it apart from competitors. "
            f"The tone should be {mood}, aiming to build connection and interest among potential customers."
        ),
        "Corporate Social Responsibility (CSR) Initiative": (
            f"Craft a post about a CSR initiative undertaken by a {company_type} company and company name {watermark_text}. "
            f"The content should describe the initiative, its impact, and why it matters to the community and the company. "
            f"The tone should be {mood} to resonate with socially conscious audiences."
        )
    }
    return prompts.get(post_type, "Invalid post type")

def get_image_prompt(occasion, company_type, post_type, visual_style, color_scheme, mood, watermark_text):
    prompts = {
        "General Announcement": (
            f"Create a striking image announcing {occasion} for a {company_type} company. "
            f"Use a {visual_style} with a {color_scheme} color scheme, conveying {mood}. {watermark_text}"
        ),
        "Product Launch": (
            f"Design a dynamic and engaging image for the launch of {occasion} by a {company_type} company and company name {watermark_text}. "
            f"The visual should reflect a {visual_style} with {color_scheme} colors and convey a sense of {mood}. "
            f"Include product imagery and branding elements that enhance the launch message."
        ),
        "Event Promotion": (
            f"Create an attractive promotional image for a {occasion} organized by a {company_type} company and company name {watermark_text}. "
            f"The image should be in a {visual_style} with a {color_scheme} palette, and should evoke feelings of {mood}. "
            f"Incorporate event-specific elements like logos, themes, or relevant visuals."
        ),
        "Special Promotion or Discount": (
            f"Design an eye-catching image to promote a {occasion} discount for a {company_type} company and company name {watermark_text}. "
            f"The image should feature a {visual_style} with a {color_scheme} that grabs attention, conveying a sense of {mood}. "
            f"Include promotional elements like discount percentages or offer details."
        ),
        "Customer Testimonials or Case Studies": (
            f"Generate an image to accompany a customer testimonial for a {company_type} company and company name {watermark_text}. "
            f"The visual should be in a {visual_style} with {color_scheme} colors, reflecting a {mood}. "
            f"Include elements that reinforce the testimonial, like customer photos or before-and-after visuals."
        ),
        "Company Milestone Celebration": (
            f"Create a celebratory image to mark {occasion} for a {company_type} company and company name {watermark_text}. "
            f"The image should have a {visual_style} with a {color_scheme} color palette, and convey {mood}. "
            f"Incorporate milestone-specific graphics like numbers, trophies, or company logos."
        ),
        "Seasonal or Holiday Greetings": (
            f"Design a festive image for a {occasion} greeting from a {company_type} company and company name {watermark_text}. "
            f"The image should reflect a {visual_style} with a {color_scheme} palette, evoking {mood}. "
            f"Include holiday-related elements like snowflakes, lights, or traditional symbols."
        ),
        "Recruitment or Job Posting": (
            f"Generate a professional and appealing image to accompany a job posting for the role of {occasion} at a {company_type} company and company name {watermark_text}. "
            f"The image should have a {visual_style} with {color_scheme} colors, and convey a {mood} that aligns with the company's culture."
        ),
        "Brand Awareness or Introduction": (
            f"Create an engaging image for a brand awareness post about a {company_type} company and company name {watermark_text}. "
            f"The visual should use a {visual_style} with a {color_scheme} palette, and convey {mood}. "
            f"Include branding elements like the logo, tagline, and key visuals that represent the company."
        ),
        "Corporate Social Responsibility (CSR) Initiative": (
            f"Design a supportive and inspiring image for a post about a CSR initiative by a {company_type} company and company name {watermark_text}. "
            f"The image should reflect a {visual_style} with {color_scheme} colors, and convey {mood}. "
            f"Include elements like project photos, community engagement, or impact statistics."
        )
    }
    return prompts.get(post_type, "Invalid post type")
