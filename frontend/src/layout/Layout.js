import React from 'react'
import { Outlet } from 'react-router-dom'

import Navbar from '../components/Navbar'

const Layout = () => {
    return (
        <div className="flex flex-col overflow-y-hidden">
            <Navbar />
            <main className="flex-grow p-4">
                <Outlet />
            </main>
        </div>
    );
};

export default Layout;
