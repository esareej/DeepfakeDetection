const fileInput = document.getElementById('file-input');
const fileChosen = document.getElementById('file-chosen');
const uploadButton = document.getElementById('upload-button');
const termsCheckbox = document.getElementById('terms-checkbox');

// Enable the upload button only if a file is chosen and the terms are agreed to
function updateUploadButtonState() {
    uploadButton.disabled = !fileInput.files.length || !termsCheckbox.checked;
}

fileInput.addEventListener('change', function() {
    fileChosen.textContent = this.files.length ? this.files[0].name : 'No file chosen';
    updateUploadButtonState();
});

termsCheckbox.addEventListener('change', updateUploadButtonState);

uploadButton.addEventListener('click', function() {
    if (fileInput.files.length > 0 && termsCheckbox.checked) {
        // Simulate the analysis process and redirect to the analysis page
        performAnalysis();
    }
});

function performAnalysis() {
    // Redirect to the analysis page
    window.location.href = 'analyze.html';
    // You would typically send the file to your server for analysis here
    // and use the server's response to update the UI or redirect to the result page.
}

// Initialize the upload button to be disabled
updateUploadButtonState();