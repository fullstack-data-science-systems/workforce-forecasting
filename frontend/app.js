const API_BASE = window.location.origin;

let forecastChart = null;
let currentForecastData = null;

const UI = {
    statusInd: document.getElementById("status-indicator"),
    statusText: document.querySelector(".status-text"),
    modelSelect: document.getElementById("model-select"),
    provinceSelect: document.getElementById("province-select"),
    stepsInput: document.getElementById("steps-input"),
    submitBtn: document.getElementById("forecast-btn"),
    loader: document.querySelector(".loader-spinner"),
    btnText: document.querySelector(".btn-primary span"),
    welcomeMsg: document.getElementById("welcome-message"),
    chartCanvas: document.getElementById("forecastChart"),
    dataTableContainer: document.querySelector(".data-table-container"),
    tableHeadRow: document.getElementById("table-head-row"),
    tableBody: document.getElementById("table-body"),
    downloadBtn: document.getElementById("download-csv-btn"),
    modelDetails: document.getElementById("model-details"),
    detailsText: document.getElementById("details-text")
};

let availableModelsData = {};

async function init() {
    try {
        await checkAPI();
        await fetchModels();
        await fetchProvinces();

        UI.modelSelect.addEventListener("change", updateModelDetails);
        document.getElementById("forecast-form").addEventListener("submit", handleForecastSubmit);
        UI.downloadBtn.addEventListener("click", downloadCSV);

        UI.submitBtn.disabled = false;
    } catch (e) {
        console.error("Init failed", e);
    }
}

async function checkAPI() {
    try {
        const res = await fetch(`${API_BASE}/health`);
        const data = await res.json();

        if (data.status === "healthy") {
            UI.statusInd.className = "status-indicator online";
            UI.statusText.textContent = "API Online";
        }
    } catch (e) {
        UI.statusInd.className = "status-indicator offline";
        UI.statusText.textContent = "API Offline";
        throw e;
    }
}

async function fetchModels() {
    const res = await fetch(`${API_BASE}/models`);
    const data = await res.json();
    availableModelsData = data.model_details;

    UI.modelSelect.innerHTML = `<option value="" disabled selected>Select a forecasting model</option>`;

    data.available_models.forEach(model => {
        const option = document.createElement("option");
        option.value = model;
        option.textContent = model.toUpperCase() + " Model";
        UI.modelSelect.appendChild(option);
    });
}

function updateModelDetails() {
    const selected = UI.modelSelect.value;
    if (selected && availableModelsData[selected]) {
        UI.modelDetails.style.display = "block";
        UI.detailsText.textContent = availableModelsData[selected];
    }
}

async function fetchProvinces() {
    const res = await fetch(`${API_BASE}/provinces`);
    const data = await res.json();

    data.provinces.forEach(prov => {
        const option = document.createElement("option");
        option.value = prov;
        option.textContent = prov;
        UI.provinceSelect.appendChild(option);
    });
}

async function handleForecastSubmit(e) {
    e.preventDefault();
    const model = UI.modelSelect.value;
    const province = UI.provinceSelect.value;
    const steps = UI.stepsInput.value;

    setLoading(true);

    try {
        const reqBody = {
            model_type: model,
            steps: parseInt(steps),
            province: province || null
        };

        const res = await fetch(`${API_BASE}/forecast`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(reqBody)
        });

        if (!res.ok) {
            const error = await res.json();
            throw new Error(error.detail || "Failed to fetch forecast");
        }

        const data = await res.json();
        currentForecastData = data;

        renderChart(data);
        renderTable(data);

        UI.welcomeMsg.style.display = "none";
        UI.chartCanvas.style.display = "block";
        UI.dataTableContainer.style.display = "flex";

    } catch (error) {
        alert("Error generating forecast: " + error.message);
    } finally {
        setLoading(false);
    }
}

function setLoading(isLoading) {
    UI.submitBtn.disabled = isLoading;
    if (isLoading) {
        UI.btnText.style.display = "none";
        UI.loader.style.display = "block";
    } else {
        UI.btnText.style.display = "block";
        UI.loader.style.display = "none";
    }
}

function renderChart(data) {
    const months = data.forecast.map(row => row.month);

    // Auto color generation
    const colors = [
        '#6c5ce7', '#00cec9', '#fdcb6e', '#ff7675', '#00b894',
        '#e84393', '#0984e3', '#d63031', '#6ab04c', '#be2edd'
    ];

    const columns = data.columns.filter(c => c !== "month");

    const datasets = columns.map((col, idx) => {
        return {
            label: col.replace(/_/g, ' '),
            data: data.forecast.map(row => row[col]),
            borderColor: colors[idx % colors.length],
            backgroundColor: `${colors[idx % colors.length]}33`,
            borderWidth: 2,
            pointRadius: 3,
            pointHoverRadius: 6,
            tension: 0.3,
            fill: false
        };
    });

    if (forecastChart) forecastChart.destroy();

    Chart.defaults.color = "#a4b0be";
    Chart.defaults.font.family = "'Inter', sans-serif";

    forecastChart = new Chart(UI.chartCanvas, {
        type: 'line',
        data: { labels: months, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        color: "#f5f6fa",
                        usePointStyle: true,
                        boxWidth: 8
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(10, 11, 16, 0.9)',
                    titleColor: '#fff',
                    bodyColor: '#a4b0be',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    padding: 10
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { maxRotation: 45, minRotation: 45 }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    title: {
                        display: true,
                        text: 'Employment Count (Thousands)'
                    }
                }
            }
        }
    });
}

function renderTable(data) {
    UI.tableHeadRow.innerHTML = "";
    UI.tableBody.innerHTML = "";

    const columns = ["month", ...data.columns.filter(c => c !== "month")];
    columns.forEach(col => {
        const th = document.createElement("th");
        th.textContent = col.replace(/_/g, ' ');
        UI.tableHeadRow.appendChild(th);
    });

    data.forecast.forEach(row => {
        const tr = document.createElement("tr");
        columns.forEach(col => {
            const td = document.createElement("td");
            let val = row[col];
            if (typeof val === 'number') {
                val = val.toFixed(1);
            }
            td.textContent = val;
            tr.appendChild(td);
        });
        UI.tableBody.appendChild(tr);
    });
}

function downloadCSV() {
    if (!currentForecastData) return;

    const columns = ["month", ...currentForecastData.columns.filter(c => c !== "month")];
    let csv = columns.join(",") + "\n";

    currentForecastData.forecast.forEach(row => {
        csv += columns.map(col => row[col]).join(",") + "\n";
    });

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.setAttribute("href", url);
    a.setAttribute("download", `forecast_${currentForecastData.model}_${currentForecastData.province_filter || 'canada'}.csv`);
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

window.addEventListener("DOMContentLoaded", init);
