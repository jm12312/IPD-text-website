import React from 'react';
import { Link } from 'react-router-dom';

const Navbar = () => {
    return (
        <header className="bg-white border-b">
            <nav className="mx-auto flex max-w-7xl items-center justify-center p-6 lg:px-8" aria-label="Global">
                <div className='flex gap-8'>
                    <Link to="/sentiment" className="text-base font-semibold leading-6 text-gray-900">
                        Sentiment
                    </Link>
                    <Link to="/emotion" className="text-base font-semibold leading-6 text-gray-900">
                        Emotion
                    </Link>
                    <a href="#" className="text-base font-semibold leading-6 text-gray-900">
                        Coming soon...
                    </a>
                </div>
            </nav>
        </header>
    );
};

export default Navbar;