document.addEventListener('DOMContentLoaded', function () {
    // Initialize Google Map
    var map = new google.maps.Map(document.getElementById('map'), {
        center: { lat: 47.6062, lng: -122.3321 }, // Seattle coordinates as an example
        zoom: 10 // Initial zoom level
    });

    // Add markers for each rental property
    var rentals = JSON.parse('<%- JSON.stringify(rentals) %>'); // Parse rentals data from EJS template
    rentals.forEach(function (rental) {
        var marker = new google.maps.Marker({
            position: { lat: rental.latitude, lng: rental.longitude },
            map: map,
            title: rental.address
        });

        // Add click listener to display info window
        marker.addListener('click', function () {
            var infoWindow = new google.maps.InfoWindow({
                content: `<div><strong>${rental.address}</strong><br>Price: $${rental.max_rent_price}</div>`
            });
            infoWindow.open(map, marker);
        });
    });
});
