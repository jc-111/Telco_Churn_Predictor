# app/dashboard_pages/prediction_tool.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.predict_churn import predict_customer_churn

def show_prediction_tool_page(model, scaler, expected_columns, thresholds, model_loaded, df):
    st.header("Churn Prediction Tool")

    if not model_loaded:
        st.error("Model not loaded. Please ensure the model is trained and saved properly.")
    else:
        st.write("Use this tool to predict whether a customer will churn based on their characteristics.")

        # Create input form with multiple columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Customer Profile")
            gender = st.selectbox('Gender', options=['Male', 'Female'])
            senior_citizen = st.selectbox('Senior Citizen', options=[0, 1])
            partner = st.selectbox('Partner', options=['Yes', 'No'])
            dependents = st.selectbox('Dependents', options=['Yes', 'No'])

        with col2:
            st.subheader("Services")
            tenure = st.slider('Tenure (months)', min_value=0, max_value=72, value=12)
            phone_service = st.selectbox('Phone Service', options=['Yes', 'No'])
            multiple_lines = st.selectbox('Multiple Lines', options=['Yes', 'No', 'No phone service'])
            internet_service = st.selectbox('Internet Service', options=['DSL', 'Fiber optic', 'No'])
            online_security = st.selectbox('Online Security', options=['Yes', 'No', 'No internet service'])
            online_backup = st.selectbox('Online Backup', options=['Yes', 'No', 'No internet service'])

        with col3:
            st.subheader("Account Details")
            contract = st.selectbox('Contract', options=['Month-to-month', 'One year', 'Two year'])
            paperless_billing = st.selectbox('Paperless Billing', options=['Yes', 'No'])
            payment_method = st.selectbox('Payment Method',
                                          options=['Electronic check', 'Mailed check',
                                                   'Bank transfer (automatic)', 'Credit card (automatic)'])
            monthly_charges = st.number_input('Monthly Charges', min_value=0.0, value=70.0, step=5.0)
            total_charges = st.number_input('Total Charges', min_value=0.0, value=monthly_charges * tenure, step=10.0)

        # Additional service options
        st.subheader("Additional Services")
        col1, col2, col3 = st.columns(3)

        with col1:
            device_protection = st.selectbox('Device Protection', options=['Yes', 'No', 'No internet service'])
        with col2:
            tech_support = st.selectbox('Tech Support', options=['Yes', 'No', 'No internet service'])
        with col3:
            streaming_tv = st.selectbox('Streaming TV', options=['Yes', 'No', 'No internet service'])
            streaming_movies = st.selectbox('Streaming Movies', options=['Yes', 'No', 'No internet service'])

        # Add threshold selection
        st.subheader("Prediction Settings")
        threshold_type = st.radio(
            "Prediction Threshold Type",
            ["Default (0.5)", "F1-Optimized", "Business-Optimized"],
            horizontal=True
        )

        # Display threshold explanation
        if threshold_type == "F1-Optimized":
            f1_threshold = thresholds.get('f1_threshold', 0.52)
            st.info(
                f"F1 threshold ({f1_threshold:.2f}) balances precision and recall to optimize overall model performance.")
        elif threshold_type == "Business-Optimized":
            business_threshold = thresholds.get('business_threshold', 0.16)
            st.info(
                f"Business threshold ({business_threshold:.2f}) maximizes ROI by considering retention costs and customer value.")

        # Create prediction button
        predict_button = st.button('Predict Churn')

        if predict_button:
            # Create input data dictionary
            input_data = {
                'customerID': 'dashboard-user',  # Add a placeholder ID
                'gender': gender,
                'SeniorCitizen': senior_citizen,
                'Partner': partner,
                'Dependents': dependents,
                'tenure': tenure,
                'PhoneService': phone_service,
                'MultipleLines': multiple_lines,
                'InternetService': internet_service,
                'OnlineSecurity': online_security,
                'OnlineBackup': online_backup,
                'DeviceProtection': device_protection,
                'TechSupport': tech_support,
                'StreamingTV': streaming_tv,
                'StreamingMovies': streaming_movies,
                'Contract': contract,
                'PaperlessBilling': paperless_billing,
                'PaymentMethod': payment_method,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }

            # Save to temporary CSV
            temp_df = pd.DataFrame([input_data])
            temp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, "temp_prediction_data.csv")
            temp_df.to_csv(temp_path, index=False)

            # Map threshold type selection to parameter
            threshold_param = 'default'
            if threshold_type == "F1-Optimized":
                threshold_param = 'f1'
            elif threshold_type == "Business-Optimized":
                threshold_param = 'business'

            # Use the function from predict_churn.py
            try:
                with st.spinner("Calculating churn probability..."):
                    results = predict_customer_churn(temp_path, threshold_type=threshold_param)
                    churn_probability = results.iloc[0]['churn_probability']
                    will_churn = results.iloc[0]['predicted_churn'] == 1
                    risk_level = results.iloc[0]['risk_level']

                # Display results
                st.header("Prediction Results")

                # Create gauge chart for probability
                fig, ax = plt.subplots(figsize=(8, 3))

                # Create gauge chart with gradient color
                risk_ranges = [(0, 0.3, '#3498db'), (0.3, 0.7, '#f39c12'), (0.7, 1.0, '#e74c3c')]
                for start, end, color in risk_ranges:
                    ax.barh(0, end - start, left=start, height=0.5, color=color, alpha=0.7)

                # Add needle
                ax.plot([churn_probability, churn_probability], [-0.1, 0.5], color='black', linewidth=2)
                ax.scatter(churn_probability, 0, s=100, color='black', zorder=5)

                # Add threshold marker
                threshold_value = 0.5
                if threshold_type == "F1-Optimized":
                    threshold_value = thresholds.get('f1_threshold', 0.52)
                elif threshold_type == "Business-Optimized":
                    threshold_value = thresholds.get('business_threshold', 0.16)

                ax.axvline(x=threshold_value, color='red', linestyle='--', alpha=0.7)
                ax.text(threshold_value, -0.4, f"Threshold: {threshold_value:.2f}",
                        ha='center', va='center', color='red', fontsize=8, rotation=90)

                # Customize chart
                ax.set_xlim(0, 1)
                ax.set_ylim(-0.5, 1)
                ax.set_xticks([0, 0.3, 0.7, 1.0])
                ax.set_xticklabels(['0%', '30%', '70%', '100%'])
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)

                plt.title(f'Churn Probability: {churn_probability:.1%} (Risk Level: {risk_level})')

                st.pyplot(fig)

                # Show message based on prediction
                if will_churn:
                    st.error(f"This customer is likely to churn with a probability of {churn_probability:.1%}")
                else:
                    st.success(f"This customer is likely to stay with a probability of {1 - churn_probability:.1%}")

                # Show key risk factors
                st.subheader("Risk Factors Analysis")

                risk_factors = []
                if contract == 'Month-to-month':
                    risk_factors.append("Month-to-month contract")
                if internet_service == 'Fiber optic':
                    risk_factors.append("Fiber optic internet service")
                if payment_method == 'Electronic check':
                    risk_factors.append("Electronic check payment method")
                if tenure < 12:
                    risk_factors.append("Customer tenure less than 1 year")
                if online_security == 'No' and internet_service != 'No':
                    risk_factors.append("No online security service")
                if tech_support == 'No' and internet_service != 'No':
                    risk_factors.append("No technical support")

                if risk_factors:
                    for factor in risk_factors:
                        st.warning(f"• {factor}")
                else:
                    st.info("No significant risk factors identified.")

                # Show recommendations
                st.subheader("Recommendations")
                if will_churn:
                    st.markdown('<div class="insights-container">', unsafe_allow_html=True)
                    st.write("Consider these strategies to retain this customer:")
                    st.write("• Offer a promotion for upgrading to a longer-term contract")
                    st.write("• Provide discounted security and support services")
                    st.write("• Consider a loyalty program or special pricing")
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="insights-container">', unsafe_allow_html=True)
                    st.write("This customer has a low churn risk, but consider these actions:")
                    st.write("• Regular check-ins to ensure continued satisfaction")
                    st.write("• Cross-sell additional services")
                    st.write("• Highlight long-term benefits of current subscription")
                    st.markdown('</div>', unsafe_allow_html=True)

                # Add similar customers analysis
                if df is not None:
                    st.subheader("Comparison to Similar Customers")

                    # Find similar customers
                    similar_customers = df[
                        (df['Contract'] == contract) &
                        (df['InternetService'] == internet_service) &
                        (df['tenure'] >= max(0, tenure - 12)) &
                        (df['tenure'] <= tenure + 12)
                        ]

                    actual_churn_rate = similar_customers['Churn'].value_counts(normalize=True).get('Yes', 0)

                    st.markdown(f"""
                    <div class="insights-container">
                    <p>Based on {len(similar_customers)} similar customers with {contract} contracts, 
                    {internet_service} internet service, and similar tenure:</p>
                    <p>Historical churn rate: <span class="summary-metric">{actual_churn_rate:.1%}</span></p>
                    <p>Predicted churn probability: <span class="summary-metric">{churn_probability:.1%}</span></p>
                    <p>This customer's risk is <span class="{'risk-high' if churn_probability > actual_churn_rate else 'risk-low'}">
                    {"higher than average" if churn_probability > actual_churn_rate else "lower than average"}</span> for this segment.</p>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())