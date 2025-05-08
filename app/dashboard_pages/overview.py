# app/dashboard_pages/overview.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def show_overview_page(df, filtered_df):
    st.header("Customer Churn Overview")

    if filtered_df is not None:
        # Add executive summary
        st.markdown('<div class="executive-summary">', unsafe_allow_html=True)
        st.markdown('<div class="subheader">Executive Summary</div>', unsafe_allow_html=True)

        total_customers = len(filtered_df)
        churn_count = filtered_df[filtered_df['Churn'] == 'Yes'].shape[0]
        churn_rate = churn_count / total_customers * 100
        monthly_revenue = filtered_df['MonthlyCharges'].sum()
        at_risk_revenue = filtered_df[filtered_df['Churn'] == 'Yes']['MonthlyCharges'].sum()

        st.markdown(f"""
        • Customer base: <span class="summary-metric">{total_customers:,}</span> customers
        • Overall churn rate: <span class="summary-metric">{churn_rate:.1f}%</span> ({churn_count:,} customers)
        • Monthly revenue: <span class="summary-metric">${monthly_revenue:,.2f}</span>
        • At-risk revenue: <span class="summary-metric">${at_risk_revenue:,.2f}</span> ({at_risk_revenue / monthly_revenue * 100:.1f}% of total)
        • High-risk segments: <span class="summary-metric">Month-to-month contracts</span> with <span class="summary-metric">Fiber optic</span> service
        • Recommended action: Target the <span class="summary-metric">{len(filtered_df[(filtered_df['Contract'] == 'Month-to-month') & (filtered_df['tenure'] < 12)]):,}</span> customers with month-to-month contracts in their first year
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Basic metrics
        col1, col2, col3 = st.columns(3)

        avg_tenure = filtered_df['tenure'].mean()
        avg_monthly = filtered_df['MonthlyCharges'].mean()

        with col1:
            st.metric("Total Customers", f"{total_customers:,}")
        with col2:
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
        with col3:
            st.metric("Average Tenure", f"{avg_tenure:.1f} months")

        st.subheader("Churn Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        churn_counts = filtered_df['Churn'].value_counts()
        ax.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', startangle=90, colors=['#3498db', '#e74c3c'])
        ax.set_title('Customer Churn Distribution')
        st.pyplot(fig)

        # Churn by key features
        st.subheader("Churn Rate by Key Factors")

        col1, col2 = st.columns(2)

        with col1:
            # Churn by contract type
            contract_churn = pd.crosstab(filtered_df['Contract'], filtered_df['Churn'], normalize='index') * 100
            fig, ax = plt.subplots(figsize=(10, 6))
            contract_churn['Yes'].plot(kind='bar', ax=ax, color='#e74c3c')
            ax.set_ylabel('Churn Rate (%)')
            ax.set_title('Churn Rate by Contract Type')
            ax.set_ylim(0, 100)
            for i, v in enumerate(contract_churn['Yes']):
                ax.text(i, v + 2, f"{v:.1f}%", ha='center')
            st.pyplot(fig)

        with col2:
            # Churn by internet service
            internet_churn = pd.crosstab(filtered_df['InternetService'], filtered_df['Churn'], normalize='index') * 100
            fig, ax = plt.subplots(figsize=(10, 6))
            internet_churn['Yes'].plot(kind='bar', ax=ax, color='#3498db')
            ax.set_ylabel('Churn Rate (%)')
            ax.set_title('Churn Rate by Internet Service')
            ax.set_ylim(0, 100)
            for i, v in enumerate(internet_churn['Yes']):
                ax.text(i, v + 2, f"{v:.1f}%", ha='center')
            st.pyplot(fig)
    else:
        st.warning("Please load the dataset to view the overview.")