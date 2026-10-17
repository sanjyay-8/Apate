const imageInput = document.getElementById("imageInput");
const uploadArea = document.getElementById("uploadArea");
const uploadTitle = document.getElementById("uploadTitle");
const uploadDescription = document.getElementById("uploadDescription");

const previewContainer = document.getElementById("previewContainer");
const imagePreview = document.getElementById("imagePreview");
const removeImage = document.getElementById("removeImage");

const analyseButton = document.getElementById("analyseButton");
const buttonText = document.getElementById("buttonText");

const errorMessage = document.getElementById("errorMessage");
const resultCard = document.getElementById("resultCard");
const resultBadge = document.getElementById("resultBadge");
const resultTitle = document.getElementById("resultTitle");
const resultMessage = document.getElementById("resultMessage");

const scoreContainer = document.getElementById("scoreContainer");
const scoreValue = document.getElementById("scoreValue");
const scoreNote = document.getElementById("scoreNote");

const API_URL = "http://148.100.98.52:8000/api/analyze";
const USE_MOCK_DATA = false;

const MAX_IMAGE_SIZE = 5 * 1024 * 1024;

let selectedFile = null;
let previewUrl = null;

function showError(message) {
    errorMessage.textContent = message;
}

function clearResults() {
    resultCard.hidden = true;
    scoreContainer.hidden = true;
    showError("");
}

function resetImage() {
    selectedFile = null;
    imageInput.value = "";

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }

    imagePreview.removeAttribute("src");
    previewContainer.hidden = true;
    uploadArea.hidden = false;

    analyseButton.disabled = true;
    buttonText.textContent = "Analyse image";

    clearResults();
}

function selectImage(file) {
    clearResults();

    if (!file) return;

    const allowedTypes = [
        "image/jpeg",
        "image/png",
        "image/webp"
    ];

    if (!allowedTypes.includes(file.type)) {
        showError("Please select a JPG, PNG or WEBP image.");
        return;
    }

    if (file.size > MAX_IMAGE_SIZE) {
        showError("The image must be smaller than 5 MB.");
        return;
    }

    selectedFile = file;

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
    }

    previewUrl = URL.createObjectURL(file);

    imagePreview.src = previewUrl;

    uploadArea.hidden = true;
    previewContainer.hidden = false;
    analyseButton.disabled = false;

    showError("");
}

imageInput.addEventListener("change", () => {
    selectImage(imageInput.files[0]);
});

removeImage.addEventListener("click", resetImage);

uploadArea.addEventListener("dragover", event => {
    event.preventDefault();
});

uploadArea.addEventListener("drop", event => {
    event.preventDefault();

    const file = event.dataTransfer.files[0];

    selectImage(file);
});

function renderResult(data) {
    resultCard.hidden = false;
    scoreContainer.hidden = true;

    if (data.status !== "success") {
        resultBadge.textContent = "UNAVAILABLE";
        resultTitle.textContent = "Analysis unavailable";
        resultMessage.textContent =
            data.message || "The image could not be analysed.";
        return;
    }

    const labels = {
        potentially_manipulated: "Potentially manipulated",
        no_manipulation_detected: "No manipulation detected",
        inconclusive: "Inconclusive result"
    };

    resultBadge.textContent = "ASSESSMENT";

    resultTitle.textContent =
        labels[data.prediction] || "Unrecognised result";

    resultMessage.textContent =
        data.message || "The model returned an assessment.";

    if (
        typeof data.score === "number" &&
        Number.isFinite(data.score) &&
        data.score >= 0 &&
        data.score <= 1 &&
        typeof data.score_meaning === "string" &&
        data.score_meaning.trim()
    ) {
        scoreContainer.hidden = false;
        scoreValue.textContent = data.score.toFixed(3);
        scoreNote.textContent = data.score_meaning;
    }
}

async function analyzeImage(file) {
    if (USE_MOCK_DATA) {
        // Development only: this is NOT a real prediction.
        await new Promise(resolve => setTimeout(resolve, 900));

        return {
            status: "success",
            prediction: "inconclusive",
            score: null,
            message:
                "Demo mode: the backend is not connected yet. " +
                "No real image analysis has been performed."
        };
    }

    const formData = new FormData();
    formData.append("image", file);

    const controller = new AbortController();

    const timeoutId = setTimeout(() => {
        controller.abort();
    }, 30000);

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            body: formData,
            signal: controller.signal
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.message || "The analysis request failed."
            );
        }

        return data;
    } finally {
        clearTimeout(timeoutId);
    }
}

analyseButton.addEventListener("click", async () => {
    if (!selectedFile) return;

    analyseButton.disabled = true;
    buttonText.textContent = "Analysing...";
    clearResults();

    try {
        const result = await analyzeImage(selectedFile);
        renderResult(result);
    } catch (error) {
        showError(
            error.name === "AbortError"
                ? "The request timed out. Please try again."
                : "Could not connect to APATE. Check the backend."
        );
    } finally {
        analyseButton.disabled = false;
        buttonText.textContent = "Analyse image";
    }
});