document.addEventListener('DOMContentLoaded', function () {
    document.querySelector('form').onsubmit = function () {

        // Create a new FormData object
        const formData = new FormData();

        // Get the text input value and append it to the FormData object
        const text = document.querySelector('#text').value;
        formData.append('type', 'text');
        formData.append('data', text);

        // Send a POST request to the server
        fetch('/', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (response.ok) {
                return response.json();
            }
            throw new Error('Network response was not ok.');
        })
        .then(data => {
            // Handle the response data
            if (data) {
                alert('No hate speech detected.');
            } else {
                alert('Hate speech detected!');
            }
        })
        .catch(error => {
            console.error('There was an error!', error);
        });

        // Prevent the default form submission
        return false;
    };
});
