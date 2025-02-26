import React, { useState } from 'react';
import axios from 'axios';

const Emotion = () => {
    const [inputText, setInputText] = useState('');
    const [output, setOutput] = useState('');
    const [probabilities, setProbabilities] = useState(null);
    const [isError, setIsError] = useState(false);

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            if (inputText.trim() === "") {
                setOutput("Input is required");
                setIsError(true);
                return;
            }

            const response = await axios.post('http://localhost:8000/predict/emotion', { input: inputText });
            
            // Extract the predicted emotion and probabilities
            const { emotion_label, probabilities } = response.data;

            setOutput(emotion_label);
            setProbabilities(probabilities); // Set the probabilities state
            setIsError(false);
        } catch (error) {
            console.error("Error fetching the ML model output:", error);
            setIsError(true);
            setOutput("Error fetching the result. Please try again.");
        }
    };

    return (
        <div className="flex flex-col items-center mt-6 bg-white p-4">
            <h2 className="max-w-full text-4xl py-8 px-4 mb-4 font-semibold">Emotions</h2>
            <form className="w-full max-w-lg mb-3" onSubmit={handleSubmit}>
                <div className="mb-4">
                    <label
                        className="block text-gray-700 text-base font-semibold mb-2"
                        htmlFor="inputText">
                        Input Text
                    </label>
                    <textarea
                        id="inputText"
                        value={inputText}
                        onChange={(e) => setInputText(e.target.value)}
                        className="shadow appearance-none border rounded text-sm w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                        rows="5"
                        placeholder="Enter your input here"
                    ></textarea>
                </div>
                <button
                    type="submit"
                    className="w-full bg-blue-500 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
                >
                    Submit
                </button>
            </form>

            {output && (
                <div className="w-full max-w-lg mt-3">
                    <h1 className="text-gray-700 text-base font-semibold mb-2">Prediction:</h1>
                    <div className={`w-full ${isError ? 'bg-red-700 text-white' : 'bg-gray-100 text-gray-800'} p-4 rounded`}>
                        <p>{output}</p>
                    </div>
                </div>
            )}

            {probabilities && (
                <div className="w-full max-w-lg mt-3">
                    <h2 className="text-gray-700 text-base font-semibold mb-2">Probabilities:</h2>
                    <div className="bg-gray-100 text-gray-800 p-4 rounded">
                        <ul>
                            {Object.entries(probabilities).map(([emotion, prob]) => (
                                <li key={emotion} className="py-1">
                                    <strong>{emotion}:</strong> {prob}
                                </li>
                            ))}
                        </ul>
                    </div>
                </div>
            )}
        </div>
    );
};

export default Emotion;
