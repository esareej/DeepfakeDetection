document.addEventListener('DOMContentLoaded', () => {
    // Setup placeholders for the multi-attentional and frequency domain analysis charts
    const multiAttentionalCtx = document.getElementById('multiAttentionalAnalysisChart').getContext('2d');
    const frequencyDomainCtx = document.getElementById('frequencyDomainAnalysisChart').getContext('2d');

    // Placeholder data for multi-attentional analysis chart
    const multiAttentionalChart = new Chart(multiAttentionalCtx, {
        type: 'line',
        data: {
            labels: ['Region 1', 'Region 2', 'Region 3', 'Region 4'],
            datasets: [{
                label: 'Attention',
                data: [0, 0, 0, 0], // Initial data, to be updated in real-time
                fill: false,
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            }]
        },
        options: {}
    });

    // Placeholder data for frequency domain analysis chart
    const frequencyDomainChart = new Chart(frequencyDomainCtx, {
        type: 'bar',
        data: {
            labels: ['Frequency 1', 'Frequency 2', 'Frequency 3', 'Frequency 4'],
            datasets: [{
                label: 'Frequency Power',
                data: [0, 0, 0, 0], // Initial data, to be updated in real-time
                backgroundColor: [
                    'rgba(255, 99, 132, 0.2)',
                    'rgba(54, 162, 235, 0.2)',
                    'rgba(255, 206, 86, 0.2)',
                    'rgba(75, 192, 192, 0.2)'
                ],
                borderColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 206, 86, 1)',
                    'rgba(75, 192, 192, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });

    // Simulate the analysis progress
    let progress = 0;
    const progressBar = document.getElementById('progress-bar');
    const percentageText = document.getElementById('percentage');
    const interval = setInterval(() => {
        progress++;
        progressBar.style.width = `${progress}%`;
        percentageText.textContent = `${progress}% Complete`;
        if (progress >= 100) {
            clearInterval(interval);
            // Update the text to indicate completion
            percentageText.textContent = "Scanning Complete";
            
            // Add a "See the Result" button
            const resultButton = document.createElement("button");
            resultButton.textContent = "See the Result";
            resultButton.id = "result-button";
            resultButton.style = "margin-top: 20px; padding: 10px 20px; font-size: 16px; cursor: pointer; border-radius: 5px; border: none; background-color: #4facfe; color: white;";

            resultButton.addEventListener("click", function() {
                window.location.href = `/result/${encodeURIComponent("{{ filename }}")}`;
            });

            // Append the button to the specific container
            document.getElementById('button-container').appendChild(resultButton);
        }
    }, 100);
});
