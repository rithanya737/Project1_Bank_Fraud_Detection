// script.js
// Handles the transaction form: on submit, calls the FastAPI /predict
// endpoint and renders the fraud probability, flag, and top SHAP
// contributing features.

const form = document.getElementById("transaction-form");
const submitBtn = document.getElementById("submit-btn");
const errorMessage = document.getElementById("error-message");
const resultCard = document.getElementById("result-card");
const resultBanner = document.getElementById("result-banner");
const resultIcon = document.getElementById("result-icon");
const resultLabel = document.getElementById("result-label");
const resultProbability = document.getElementById("result-probability");
const resultThreshold = document.getElementById("result-threshold");
const featuresList = document.getElementById("features-list");

// Default the timestamp field to "now" for convenience
const timestampInput = document.getElementById("timestamp");
const now = new Date();
now.setSeconds(0, 0);
timestampInput.value = now.toISOString().slice(0, 16);

function readableFeatureName(name) {
  // Turn e.g. "merchant_category_gambling" or "device_risk_score" into
  // a friendlier label for display.
  return name
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function renderResult(data) {
  featuresList.innerHTML = "";

  const isFraud = data.fraud_flag;
  resultBanner.className = "result-banner " + (isFraud ? "fraud" : "safe");
  resultIcon.textContent = isFraud ? "⚠️" : "✅";
  resultLabel.textContent = isFraud ? "Likely Fraudulent" : "Likely Legitimate";
  resultProbability.textContent = `Fraud probability: ${(data.fraud_probability * 100).toFixed(2)}%`;
  resultThreshold.textContent = data.threshold_used.toFixed(3);

  data.top_contributing_features.forEach((f) => {
    const li = document.createElement("li");
    li.className = "feature-item";

    const increases = f.direction === "increases_fraud_risk";
    li.innerHTML = `
      <div>
        <div class="feature-name">${readableFeatureName(f.feature)}</div>
        <div class="feature-value">value = ${Number(f.value).toFixed(3)}, SHAP = ${f.shap_value.toFixed(3)}</div>
      </div>
      <span class="feature-impact ${increases ? "increases" : "decreases"}">
        ${increases ? "↑ Raises Risk" : "↓ Lowers Risk"}
      </span>
    `;
    featuresList.appendChild(li);
  });

  resultCard.hidden = false;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorMessage.hidden = true;
  resultCard.hidden = true;

  const apiUrl = document.getElementById("api-url").value.replace(/\/$/, "");

  const payload = {
    amount: parseFloat(document.getElementById("amount").value),
    timestamp: document.getElementById("timestamp").value,
    transaction_type: document.getElementById("transaction_type").value,
    merchant_category: document.getElementById("merchant_category").value,
    country: document.getElementById("country").value,
    device_risk_score: parseFloat(document.getElementById("device_risk_score").value),
    ip_risk_score: parseFloat(document.getElementById("ip_risk_score").value),
  };

  submitBtn.disabled = true;
  submitBtn.textContent = "Checking...";

  try {
    const response = await fetch(`${apiUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`API error (${response.status}): ${errText}`);
    }

    const data = await response.json();
    renderResult(data);
  } catch (err) {
    errorMessage.textContent = `Could not get a prediction: ${err.message}. Is the backend running at ${apiUrl}?`;
    errorMessage.hidden = false;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Check Transaction";
  }
});
