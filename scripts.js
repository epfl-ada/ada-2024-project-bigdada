

// Adding an event to detect when the page is scrolled
window.onscroll = function() {
    var topBar = document.querySelector('.top-bar');
    if (window.scrollY > 0) {
        topBar.classList.add('scrolled');  // Adds the .scrolled class when scrolling
    } else {
        topBar.classList.remove('scrolled'); // Removes the class when at the top of the page
    }
};

function toggleMenu() {
    const menu = document.getElementById('menu');
    const overlay = document.getElementById('menu-overlay');

    // Toggle the menu and overlay
    const isVisible = menu.classList.toggle('show');
    overlay.style.display = isVisible ? 'block' : 'none';
}

// Close the menu when clicking on the overlay
document.getElementById('menu-overlay').addEventListener('click', () => {
    const menu = document.getElementById('menu');
    const overlay = document.getElementById('menu-overlay');

    menu.classList.remove('show');
    overlay.style.display = 'none';
});

// Optional: Close the menu when clicking on a menu link
document.querySelectorAll('.menu a').forEach(link => {
    link.addEventListener('click', () => {
        const menu = document.getElementById('menu');
        const overlay = document.getElementById('menu-overlay');

        menu.classList.remove('show');
        overlay.style.display = 'none';
    });
});


document.addEventListener("DOMContentLoaded", function () {
    function loadSection(sectionId, filePath) {
        fetch(filePath)
            .then(response => response.text())
            .then(data => document.getElementById(sectionId).innerHTML = data)
            .catch(error => console.log('Erreur de chargement:', error));
    }

    loadSection('introduction', 'sections/introduction.html');
    
});

let currentPage = 1;
const totalPages = 2; 


function changePage(direction) {
    currentPage += direction;
    
    // Si la page actuelle dépasse les limites, on recommence depuis le début ou on va à la fin
    if (currentPage < 1) {
        currentPage = totalPages;
    } else if (currentPage > totalPages) {
        currentPage = 1;
    }
    
    // Cache toutes les pages et affiche la page correspondante
    for (let i = 1; i <= totalPages; i++) {
        document.getElementById('page' + i).style.display = 'none';
    }
    
    document.getElementById('page' + currentPage).style.display = 'flex';
}

// Initialiser la première page
document.addEventListener('DOMContentLoaded', function() {
    changePage(0);
});



