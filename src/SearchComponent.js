import React, { useState } from "react";
import axios from "axios";

function SearchComponent() {
    const [query, setQuery] = useState("");
    const [results, setResults] = useState([]);
    const [loading, setLoading] = useState(false);

    const handleSearch = async () => {
        setLoading(true);
        try {
            const response = await axios.get(`http://localhost:5000/search?query=${query}`);
            setResults(response.data);
        } catch (error) {
            console.error("Error fetching search results:", error);
        }
        setLoading(false);
    };

    return (
        <div>
            <h1>Search for Documents</h1>
            <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Enter your search query"
            />
            <button onClick={handleSearch}>Search</button>
            
            {loading && <p>Loading...</p>}

            <div>
                <h3>Results:</h3>
                {results && results.length > 0 ? (
                    <ul>
                        {results.map((result, index) => (
                            <li key={index}>{result}</li>
                        ))}
                    </ul>
                ) : (
                    <p>No results found</p>
                )}
            </div>
        </div>
    );
}

export default SearchComponent;
