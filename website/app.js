const express = require('express');
const path = require('path');
const sqlite3 = require('sqlite3').verbose();
const fs = require('fs');
const csv = require('csv-parser');

const app = express();
const dbFile = 'rentals.db';

app.set('view engine', 'ejs');
app.use(express.static(path.join(__dirname, 'views')));

function createDatabase(callback) {
    const db = new sqlite3.Database(dbFile);

    db.serialize(() => {
        db.run(`CREATE TABLE IF NOT EXISTS rentals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            property_id TEXT,
            url TEXT,
            property_type TEXT,
            address TEXT,
            city TEXT,
            state TEXT,
            zip_code TEXT,
            country_code TEXT,
            latitude REAL,
            longitude REAL,
            rental_id TEXT,
            max_beds INTEGER,
            max_baths REAL,
            max_square_feet REAL,
            max_rent_price REAL,
            description TEXT
        )`);

        db.run(`DELETE FROM rentals`);

        fs.createReadStream('../washington_rentals.csv')
            .pipe(csv())
            .on('data', (row) => {
                db.run(`INSERT INTO rentals (
                    property_id, url, property_type, address, city, state, zip_code,
                    country_code, latitude, longitude, rental_id, max_beds, max_baths,
                    max_square_feet, max_rent_price, description
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`, [
                    row['Property ID'], row['URL'], row['Property Type'], row['Address'],
                    row['City'], row['State'], row['ZIP Code'], row['Country Code'],
                    row['Latitude'], row['Longitude'], row['Rental ID'], row['Max Beds'],
                    row['Max Baths'], row['Max Square Feet'], row['Max Rent Price'], row['Description']
                ]);
            })
            .on('end', () => {
                console.log('CSV file successfully processed and data inserted into the database');
                if (callback) callback();
            });
    });

    return db;
}

const db = createDatabase(() => {
    const PORT = process.env.PORT || 3000;
    app.listen(PORT, () => {
        console.log(`Server is running on port ${PORT}`);
    });
});

app.get('/', (req, res) => {
    db.all(`SELECT * FROM rentals`, (err, rows) => {
        if (err) {
            console.error(err);
            res.status(500).send('Database error');
            return;
        }

        // Pass API key as a string to the index template
        res.render('index', { rentals: rows, apiKey: process.env.GOOGLE_MAPS_API_KEY });
    });
});

app.get('/search', (req, res) => {
    const city = req.query.city?.trim().toUpperCase();
    const zipCode = req.query.zip_code?.trim();
    const minBeds = req.query.min_beds;
    const minBaths = req.query.min_baths;
    const minSquareFeet = req.query.min_square_feet;

    let query = `SELECT * FROM rentals WHERE 1=1`;
    const params = [];

    if (city) {
        query += ` AND UPPER(city) LIKE ?`;
        params.push(`%${city}%`);
    }

    if (zipCode) {
        query += ` AND zip_code LIKE ?`;
        params.push(`%${zipCode}%`);
    }

    if (minBeds) {
        query += ` AND max_beds >= ?`;
        params.push(minBeds);
    }

    if (minBaths) {
        query += ` AND max_baths >= ?`;
        params.push(minBaths);
    }

    if (minSquareFeet) {
        query += ` AND max_square_feet >= ?`;
        params.push(minSquareFeet);
    }

    db.all(query, params, (err, rows) => {
        if (err) {
            console.error(err);
            res.status(500).send('Database error');
            return;
        }

        res.render('index', { rentals: rows, apiKey: process.env.GOOGLE_MAPS_API_KEY });
    });
});
