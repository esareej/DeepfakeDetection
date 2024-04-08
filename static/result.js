document.addEventListener('DOMContentLoaded', () => {
    const backButton = document.querySelector('.back-button');
    backButton.addEventListener('click', function() {
        window.location.href = "{{ url_for('home') }}";
    });

    // Retrieve the score and result from the rendered template
    const score = parseFloat(document.getElementById('score').textContent);
    const result = document.getElementById('final-result-value').textContent;

    // Update the score on the page
    const scoreElement = document.getElementById('score');
    scoreElement.textContent = score.toFixed(2);

    // Determine the prediction and update the page accordingly
    const finalResultElement = document.getElementById('final-result-value');

    if (result === 'fake') {
        finalResultElement.style.color = '#e53e3e'; // Red color for fake
    } else if (result === 'real') {
        finalResultElement.style.color = '#48bb78'; // Green color for real
    } else {
        finalResultElement.style.color = '#d69e2e'; // Yellow color for unsure
    }
});
