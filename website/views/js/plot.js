document.addEventListener('DOMContentLoaded', function () {
    // google maps
    var map = new google.maps.Map(document.getElementById('map'), {
        center: { lat: 47.6062, lng: -122.3321 }, 
        zoom: 10 
    });

    // rental properties
    var rentals = JSON.parse('<%- JSON.stringify(rentals) %>');
    rentals.forEach(function (rental) {
        var marker = new google.maps.Marker({
            position: { lat: rental.latitude, lng: rental.longitude },
            map: map,
            title: rental.address
        });

        marker.addListener('click', function () {
            var infoWindow = new google.maps.InfoWindow({
                content: `<div><strong>${rental.address}</strong><br>Price: $${rental.max_rent_price}</div>`
            });
            infoWindow.open(map, marker);
        });
    });
});
