const fileInput = document.getElementById('file-input');
const fileChosen = document.getElementById('file-chosen');
const uploadButton = document.getElementById('upload-button');
const termsCheckbox = document.getElementById('terms-checkbox');

document.addEventListener('DOMContentLoaded', function() {
    var menuIcon = document.querySelector('.menu-icon');
    var navMenu = document.querySelector('.nav-menu');
    
    // Toggle the menu on icon click
    menuIcon.addEventListener('click', function() {
        // Checks if the menu is already shown
        if (navMenu.style.display === 'block') {
            navMenu.style.display = 'none'; // Hide the menu
        } else {
            navMenu.style.display = 'block'; // Show the menu
        }
    });
});

function openNav() {
    document.getElementById("mySidenav").style.width = "250px";
    document.getElementById("myOverlay").style.display = "block"; // Show the overlay
}

function closeNav() {
    document.getElementById("mySidenav").style.width = "0";
    document.getElementById("myOverlay").style.display = "none"; // Hide the overlay
}


// Function to update the state of the upload button
function updateUploadButtonState() {
    // Now we only update the button's visual state, not its disabled property
    if (!fileInput.files.length || !termsCheckbox.checked) {
        uploadButton.classList.add("disabled"); // Add a 'disabled' class for styling
    } else {
        uploadButton.classList.remove("disabled"); // Remove the 'disabled' class
    }
}

fileInput.addEventListener('change', function() {
    fileChosen.textContent = this.files.length ? this.files[0].name : 'No file chosen';
    updateUploadButtonState();
});

termsCheckbox.addEventListener('change', updateUploadButtonState);

uploadButton.addEventListener('click', function(event) {
    // Prevent the default form submission if the terms are not agreed to
    if (!termsCheckbox.checked) {
        alert("You must agree to the terms in order to proceed.");
        event.preventDefault(); // Prevent the button action if terms are not agreed
    } else if (fileInput.files.length > 0) {
        performAnalysis();
    }
});

function performAnalysis() {
    // Redirect to the result page
    window.location.href = "{{ url_for('result') }}";
}

// Initialize the upload button to be disabled
updateUploadButtonState();
