import { Routes, Route } from 'react-router-dom'
import { Navigate } from 'react-router-dom';
import Sentiment from "./pages/Sentiment.jsx";
import Navbar from './components/Navbar.js';
import Emotion from './pages/Emotion.jsx';
import Layout from './layout/Layout.js';

function App() {
  return (
    <Routes>
      <Route element={<Layout />} >
        <Route index path="/sentiment" element={<Sentiment />} />
        <Route path="/emotion" element={<Emotion />} />
        <Route path="/" element={<Sentiment />} />
      </Route>
    </Routes>

  );
}

export default App;