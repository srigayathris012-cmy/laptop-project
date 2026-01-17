import gradio as gr
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from datetime import datetime

# Load data
def load_data():
    df = pd.read_csv("laptop.csv")
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    
    df_display = df.copy()
    df["Price"] = df["Price"].str.replace("₹", "").str.replace(",", "").astype(int)
    df["Ram_GB"] = df["Ram"].str.extract(r"(\d+)").fillna(0).astype(int)
    df["SSD_GB"] = df["SSD"].str.extract(r"(\d+)").fillna(0).astype(int)
    
    def graphics_flag(x):
        return 0 if "Intel" in str(x) or "UHD" in str(x) else 1
    
    df["Graphics_Flag"] = df["Graphics"].apply(graphics_flag)
    df["Rating"] = df["Rating"].fillna(df["Rating"].mean())
    
    return df, df_display

df, df_display = load_data()

# Train model
def train_model(df):
    X = df[["Price", "Ram_GB", "SSD_GB", "Rating", "Graphics_Flag"]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    knn = NearestNeighbors(n_neighbors=5, metric="euclidean")
    knn.fit(X_scaled)
    return knn, scaler

knn, scaler = train_model(df)

# Feature 1: Use Case Based Recommendations
def recommend_by_usecase(use_case):
    use_case_specs = {
        "🎮 Gaming": {"min_ram": 16, "min_ssd": 512, "graphics": 1, "min_budget": 70000},
        "💻 Programming": {"min_ram": 16, "min_ssd": 512, "graphics": 0, "min_budget": 50000},
        "🎨 Video Editing": {"min_ram": 32, "min_ssd": 1024, "graphics": 1, "min_budget": 100000},
        "📚 Student": {"min_ram": 8, "min_ssd": 256, "graphics": 0, "min_budget": 30000},
        "💼 Business": {"min_ram": 8, "min_ssd": 512, "graphics": 0, "min_budget": 50000}
    }
    
    specs = use_case_specs.get(use_case, use_case_specs["📚 Student"])
    
    filtered = df[
        (df["Ram_GB"] >= specs["min_ram"]) &
        (df["SSD_GB"] >= specs["min_ssd"]) &
        (df["Graphics_Flag"] == specs["graphics"]) &
        (df["Price"] >= specs["min_budget"])
    ].head(5)
    
    if len(filtered) == 0:
        return "❌ No laptops found matching these criteria. Try different settings."
    
    results = f"## 🎯 Top Laptops for {use_case}\n\n"
    for idx, row in filtered.iterrows():
        results += f"### {df_display.iloc[idx]['Model']}\n"
        results += f"💰 {df_display.iloc[idx]['Price']} | "
        results += f"💾 {df_display.iloc[idx]['Ram']} | "
        results += f"💿 {df_display.iloc[idx]['SSD']}\n"
        results += f"🎮 {df_display.iloc[idx]['Graphics']} | "
        results += f"⭐ {df_display.iloc[idx]['Rating']}\n\n"
    
    return results

# Feature 2: Compare Laptops
def compare_laptops(laptop1_idx, laptop2_idx):
    if laptop1_idx == laptop2_idx:
        return "⚠️ Please select different laptops to compare"
    
    l1 = df_display.iloc[laptop1_idx]
    l2 = df_display.iloc[laptop2_idx]
    
    comparison = f"""
# 📊 Laptop Comparison

## 🔵 Laptop 1: {l1['Model']}
- 💰 Price: {l1['Price']}
- 💾 RAM: {l1['Ram']}
- 💿 SSD: {l1['SSD']}
- 🎮 Graphics: {l1['Graphics']}
- ⭐ Rating: {l1['Rating']}

## 🔴 Laptop 2: {l2['Model']}
- 💰 Price: {l2['Price']}
- 💾 RAM: {l2['Ram']}
- 💿 SSD: {l2['SSD']}
- 🎮 Graphics: {l2['Graphics']}
- ⭐ Rating: {l2['Rating']}

---

## 🏆 Winner Analysis:

"""
    
    # Price comparison
    price1 = int(l1['Price'].replace('₹', '').replace(',', ''))
    price2 = int(l2['Price'].replace('₹', '').replace(',', ''))
    
    if price1 < price2:
        comparison += f"💰 **Better Value:** Laptop 1 (₹{price2 - price1:,} cheaper)\n"
    else:
        comparison += f"💰 **Better Value:** Laptop 2 (₹{price1 - price2:,} cheaper)\n"
    
    # RAM comparison
    ram1 = df.iloc[laptop1_idx]['Ram_GB']
    ram2 = df.iloc[laptop2_idx]['Ram_GB']
    
    if ram1 > ram2:
        comparison += f"💾 **More RAM:** Laptop 1 ({ram1} GB vs {ram2} GB)\n"
    elif ram2 > ram1:
        comparison += f"💾 **More RAM:** Laptop 2 ({ram2} GB vs {ram1} GB)\n"
    else:
        comparison += f"💾 **RAM:** Same ({ram1} GB)\n"
    
    # Rating comparison
    if l1['Rating'] > l2['Rating']:
        comparison += f"⭐ **Higher Rating:** Laptop 1 ({l1['Rating']} vs {l2['Rating']})\n"
    elif l2['Rating'] > l1['Rating']:
        comparison += f"⭐ **Higher Rating:** Laptop 2 ({l2['Rating']} vs {l1['Rating']})\n"
    else:
        comparison += f"⭐ **Rating:** Same ({l1['Rating']})\n"
    
    return comparison

# Feature 3: Budget Optimizer
def optimize_budget(max_budget, priority):
    filtered = df[df['Price'] <= max_budget].copy()
    
    if len(filtered) == 0:
        return f"❌ No laptops found under ₹{max_budget:,}. Try increasing your budget."
    
    if priority == "Performance":
        filtered = filtered.sort_values(['Ram_GB', 'SSD_GB', 'Graphics_Flag'], ascending=False)
    elif priority == "Rating":
        filtered = filtered.sort_values('Rating', ascending=False)
    else:  # Value for Money
        filtered['value_score'] = (filtered['Ram_GB'] + filtered['SSD_GB']/100 + filtered['Graphics_Flag']*10) / filtered['Price'] * 100000
        filtered = filtered.sort_values('value_score', ascending=False)
    
    results = f"## 🎯 Best Laptops Under ₹{max_budget:,} (Priority: {priority})\n\n"
    
    for idx, row in filtered.head(5).iterrows():
        results += f"### {df_display.iloc[idx]['Model']}\n"
        results += f"💰 {df_display.iloc[idx]['Price']} | "
        results += f"💾 {df_display.iloc[idx]['Ram']} | "
        results += f"💿 {df_display.iloc[idx]['SSD']}\n"
        results += f"🎮 {df_display.iloc[idx]['Graphics']} | "
        results += f"⭐ {df_display.iloc[idx]['Rating']}\n\n"
    
    return results

# Feature 4: AI Chatbot Style Recommendations
def chatbot_recommend(question1, question2, question3):
    # Gaming frequency
    if question1 == "Daily":
        min_graphics = 1
        min_ram = 16
    elif question1 == "Occasionally":
        min_graphics = 1
        min_ram = 8
    else:
        min_graphics = 0
        min_ram = 8
    
    # Budget
    if question2 == "Under ₹50k":
        max_price = 50000
    elif question2 == "₹50k - ₹80k":
        max_price = 80000
    else:
        max_price = 150000
    
    # Storage needs
    if question3 == "Heavy (1TB+)":
        min_ssd = 1024
    elif question3 == "Medium (512GB)":
        min_ssd = 512
    else:
        min_ssd = 256
    
    filtered = df[
        (df['Graphics_Flag'] >= min_graphics) &
        (df['Ram_GB'] >= min_ram) &
        (df['Price'] <= max_price) &
        (df['SSD_GB'] >= min_ssd)
    ].head(5)
    
    if len(filtered) == 0:
        return "❌ No laptops match your preferences. Try adjusting your requirements."
    
    results = "## 🤖 AI Recommendations Based on Your Answers\n\n"
    
    for idx, row in filtered.iterrows():
        results += f"### {df_display.iloc[idx]['Model']}\n"
        results += f"💰 {df_display.iloc[idx]['Price']} | "
        results += f"💾 {df_display.iloc[idx]['Ram']} | "
        results += f"💿 {df_display.iloc[idx]['SSD']}\n"
        results += f"🎮 {df_display.iloc[idx]['Graphics']} | "
        results += f"⭐ {df_display.iloc[idx]['Rating']}\n\n"
    
    return results

# Feature 5: Price Alert (Simulated)
def create_price_alert(laptop_name, target_price):
    matching = df_display[df_display['Model'].str.contains(laptop_name, case=False, na=False)]
    
    if len(matching) == 0:
        return f"❌ No laptop found with name: {laptop_name}"
    
    laptop = matching.iloc[0]
    current_price = int(laptop['Price'].replace('₹', '').replace(',', ''))
    
    if target_price >= current_price:
        return f"✅ Good news! {laptop['Model']} is already at ₹{current_price:,}, which is below your target of ₹{target_price:,}!"
    
    return f"""
🔔 Price Alert Created!

**Laptop:** {laptop['Model']}
**Current Price:** ₹{current_price:,}
**Target Price:** ₹{target_price:,}
**Difference:** ₹{current_price - target_price:,} above target

💡 You'll be notified when the price drops! (Simulated - in real app, this would send email/SMS)
"""

# Create Gradio Interface
with gr.Blocks(title="💻 Smart Laptop Finder", theme=gr.themes.Soft(primary_hue="purple")) as demo:
    
    gr.Markdown("""
    # 💻 Smart Laptop Finder Pro
    ### AI-Powered Laptop Recommendations with Advanced Features
    """)
    
    with gr.Tabs():
        
        # Tab 1: Use Case Recommendations
        with gr.Tab("🎯 Find by Use Case"):
            gr.Markdown("## Tell us what you need the laptop for:")
            usecase_input = gr.Radio(
                choices=["🎮 Gaming", "💻 Programming", "🎨 Video Editing", "📚 Student", "💼 Business"],
                label="Select Your Primary Use Case",
                value="📚 Student"
            )
            usecase_btn = gr.Button("🚀 Get Recommendations", variant="primary")
            usecase_output = gr.Markdown()
            
            usecase_btn.click(recommend_by_usecase, inputs=usecase_input, outputs=usecase_output)
        
        # Tab 2: Compare Laptops
        with gr.Tab("⚖️ Compare Laptops"):
            gr.Markdown("## Compare Two Laptops Side-by-Side")
            
            laptop_choices = [f"{i}: {model}" for i, model in enumerate(df_display['Model'].head(50))]
            
            with gr.Row():
                laptop1 = gr.Dropdown(choices=list(range(50)), label="Select Laptop 1", value=0)
                laptop2 = gr.Dropdown(choices=list(range(50)), label="Select Laptop 2", value=1)
            
            compare_btn = gr.Button("📊 Compare", variant="primary")
            compare_output = gr.Markdown()
            
            compare_btn.click(compare_laptops, inputs=[laptop1, laptop2], outputs=compare_output)
        
        # Tab 3: Budget Optimizer
        with gr.Tab("💰 Budget Optimizer"):
            gr.Markdown("## Find the Best Laptop Within Your Budget")
            
            budget_input = gr.Slider(
                minimum=20000,
                maximum=200000,
                value=60000,
                step=5000,
                label="Maximum Budget (₹)"
            )
            priority_input = gr.Radio(
                choices=["Performance", "Rating", "Value for Money"],
                label="What's most important to you?",
                value="Value for Money"
            )
            
            budget_btn = gr.Button("🔍 Find Best Options", variant="primary")
            budget_output = gr.Markdown()
            
            budget_btn.click(optimize_budget, inputs=[budget_input, priority_input], outputs=budget_output)
        
        # Tab 4: AI Chatbot
        with gr.Tab("🤖 AI Assistant"):
            gr.Markdown("## Answer a Few Questions, Get Perfect Recommendations")
            
            q1 = gr.Radio(
                choices=["Never", "Occasionally", "Daily"],
                label="❓ How often will you game on this laptop?",
                value="Occasionally"
            )
            q2 = gr.Radio(
                choices=["Under ₹50k", "₹50k - ₹80k", "Above ₹80k"],
                label="❓ What's your budget range?",
                value="₹50k - ₹80k"
            )
            q3 = gr.Radio(
                choices=["Light (256GB)", "Medium (512GB)", "Heavy (1TB+)"],
                label="❓ How much storage do you need?",
                value="Medium (512GB)"
            )
            
            chatbot_btn = gr.Button("🎯 Get My Recommendations", variant="primary")
            chatbot_output = gr.Markdown()
            
            chatbot_btn.click(chatbot_recommend, inputs=[q1, q2, q3], outputs=chatbot_output)
        
        # Tab 5: Price Alerts
        with gr.Tab("🔔 Price Alerts"):
            gr.Markdown("## Set Price Alerts for Your Favorite Laptops")
            
            alert_laptop = gr.Textbox(
                label="Laptop Model Name",
                placeholder="e.g., HP Pavilion, Dell Inspiron..."
            )
            alert_price = gr.Number(
                label="Target Price (₹)",
                value=50000
            )
            
            alert_btn = gr.Button("🔔 Create Alert", variant="primary")
            alert_output = gr.Markdown()
            
            alert_btn.click(create_price_alert, inputs=[alert_laptop, alert_price], outputs=alert_output)
    
    gr.Markdown("""
    ---
    💡 **Pro Features:** Use Case Matching | Side-by-Side Comparison | Budget Optimization | AI Assistant | Price Alerts
    
    🤖 Powered by Machine Learning | Built with Gradio
    """)

if __name__ == "__main__":
    demo.launch(share=True)
