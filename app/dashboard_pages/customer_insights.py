# app/dashboard_pages/customer_insights.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.churn_utils import get_artifacts_path


def show_customer_insights_page(df, filtered_df):
    st.header("Customer Insights")

    if filtered_df is not None:
        st.subheader("Feature Importances")

        # Feature importance from saved image
        artifacts_path = get_artifacts_path()
        feature_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'artifacts', 'feature_importance.png')
        # Also check in artifacts directory
        if not os.path.exists(feature_path):
            feature_path = os.path.join(artifacts_path, 'feature_importance.png')

        if os.path.exists(feature_path):
            st.image(feature_path, caption='Top 10 Feature Importances')

            # Check for detailed feature importance CSV
            feature_importance_csv = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models',
                                                  'feature_importance.csv')
            if os.path.exists(feature_importance_csv):
                feature_imp_df = pd.read_csv(feature_importance_csv)

                # Display top 15 features
                st.subheader("Detailed Feature Importance")
                top_features = feature_imp_df.head(15)

                # Create horizontal bar chart
                fig, ax = plt.subplots(figsize=(10, 8))
                bars = ax.barh(top_features['Feature'], top_features['Importance'], color='#1E88E5')
                ax.set_xlabel('Importance')
                ax.set_title('Top 15 Features by Importance')

                # Add value labels
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                            f"{width:.4f}", ha='left', va='center')

                st.pyplot(fig)

                # Show the data table
                st.dataframe(top_features)
        else:
            st.warning("Feature importance plot not found. Please run 'run.py' to generate it.")

        st.subheader("Tenure vs. Monthly Charges")

        # Create scatter plot of tenure vs. monthly charges, colored by churn
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = {'Yes': '#e74c3c', 'No': '#3498db'}
        for churn in ['Yes', 'No']:
            mask = filtered_df['Churn'] == churn
            ax.scatter(filtered_df.loc[mask, 'tenure'], filtered_df.loc[mask, 'MonthlyCharges'],
                       c=colors[churn], label=churn, alpha=0.6)

        ax.set_xlabel('Tenure (months)')
        ax.set_ylabel('Monthly Charges ($)')
        ax.set_title('Tenure vs. Monthly Charges by Churn')
        ax.legend(title='Churn')
        st.pyplot(fig)

        # Customer segments (if available)
        segments_path = os.path.join(artifacts_path, 'customer_segments.png')
        if not os.path.exists(segments_path):
            segments_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'artifacts', 'customer_segments.png')

        if os.path.exists(segments_path):
            st.subheader("Customer Segments")
            st.image(segments_path, caption='Customer Segments')

            csv_path = os.path.join(artifacts_path, 'customer_segments.csv')
            if not os.path.exists(csv_path):
                csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'artifacts', 'customer_segments.csv')

            if os.path.exists(csv_path):
                segments_df = pd.read_csv(csv_path)
                st.dataframe(segments_df)

        st.subheader("Additional Insights")

        col1, col2 = st.columns(2)

        with col1:
            # Churn by payment method
            payment_churn = pd.crosstab(filtered_df['PaymentMethod'], filtered_df['Churn'], normalize='index') * 100
            fig, ax = plt.subplots(figsize=(10, 6))
            payment_churn['Yes'].plot(kind='bar', ax=ax, color='#9b59b6')
            ax.set_ylabel('Churn Rate (%)')
            ax.set_title('Churn Rate by Payment Method')
            ax.set_ylim(0, 100)
            plt.xticks(rotation=45, ha='right')
            st.pyplot(fig)

        with col2:
            # Churn by tenure groups
            filtered_df['tenure_group'] = pd.cut(filtered_df['tenure'], bins=[0, 12, 24, 36, 60, 72],
                                                 labels=['0-12', '13-24', '25-36', '37-60', '61-72'])
            tenure_churn = pd.crosstab(filtered_df['tenure_group'], filtered_df['Churn'], normalize='index') * 100
            fig, ax = plt.subplots(figsize=(10, 6))
            tenure_churn['Yes'].plot(kind='bar', ax=ax, color='#2ecc71')
            ax.set_ylabel('Churn Rate (%)')
            ax.set_title('Churn Rate by Tenure Group (months)')
            ax.set_ylim(0, 100)
            for i, v in enumerate(tenure_churn['Yes']):
                ax.text(i, v + 2, f"{v:.1f}%", ha='center')
            st.pyplot(fig)

        # Add Customer Lifetime Value analysis
        st.subheader("Customer Lifetime Value Analysis")

        # Create a simple CLV calculation
        contract_data = []
        for contract in filtered_df['Contract'].unique():
            contract_df = filtered_df[filtered_df['Contract'] == contract]
            churn_rate = contract_df['Churn'].value_counts(normalize=True).get('Yes', 0)
            avg_monthly = contract_df['MonthlyCharges'].mean()
            avg_tenure = contract_df['tenure'].mean()

            # Calculate expected lifetime and CLV
            expected_lifetime = 1 / churn_rate if churn_rate > 0 else 60  # Cap at 5 years (60 months) if churn rate is 0
            clv = avg_monthly * expected_lifetime

            contract_data.append({
                'Contract': contract,
                'Avg Monthly Revenue': f"${avg_monthly:.2f}",
                'Churn Rate': f"{churn_rate:.1%}",
                'Exp. Lifetime (months)': f"{expected_lifetime:.1f}",
                'Customer Lifetime Value': f"${clv:.2f}"
            })

        # Display CLV data
        clv_df = pd.DataFrame(contract_data)
        st.dataframe(clv_df, hide_index=True)

        st.markdown("""
        <div class="insights-container">
        Customer Lifetime Value (CLV) is calculated as Average Monthly Revenue × Expected Lifetime, where Expected Lifetime = 1/Churn Rate.

        This shows the dramatic difference in customer value across contract types. Month-to-month customers have much lower lifetime value due to high churn rates,
        while customers on longer contracts provide significantly more value over time.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("Please load the dataset to view customer insights.")