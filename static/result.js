document.addEventListener('DOMContentLoaded', () => {
    const backButton = document.getElementById('back-button');
    backButton.addEventListener('click', function() {
        window.location.href = '/';
    });

    // Retrieve the result and confidence from the HTML elements
    const resultElement = document.getElementById('result');
    const confidenceElement = document.getElementById('confidence');

    const result = resultElement.textContent.trim();
    const confidence = parseFloat(confidenceElement.textContent);

    // Update the score on the page
    const scoreElement = document.getElementById('score');
    scoreElement.textContent = confidence.toFixed(2);

    // Determine the prediction and update the page accordingly
    const predictionElement = document.getElementById('prediction');
    const finalResultElement = document.getElementById('final-result-value');

    if (result === 'fake') {
        predictionElement.textContent = 'Fake';
        finalResultElement.textContent = 'Fake';
        finalResultElement.style.color = '#e53e3e'; // Red color for fake
    } else if (result === 'real') {
        predictionElement.textContent = 'Real';
        finalResultElement.textContent = 'Real';
        finalResultElement.style.color = '#48bb78'; // Green color for real
    } else {
        predictionElement.textContent = 'Uncertain';
        finalResultElement.textContent = 'Uncertain';
        finalResultElement.style.color = '#d69e2e'; // Yellow color for unsure
    }
});