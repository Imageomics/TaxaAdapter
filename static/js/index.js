// Ensure our code runs after the DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM fully loaded, initializing...');
    
    // Check for click events on the navbar burger icon - vanilla JS
    const navbarBurgers = document.querySelectorAll(".navbar-burger");
    navbarBurgers.forEach(function(burger) {
      burger.addEventListener('click', function() {
        // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
        burger.classList.toggle("is-active");
        const navbarMenu = document.querySelector(".navbar-menu");
        if (navbarMenu) {
          navbarMenu.classList.toggle("is-active");
        }
      });
    });

    // Back to top button functionality - vanilla JS
    const backToTopButton = document.getElementById("back-to-top");
    
    // Show/hide back to top button based on scroll position
    let scrollTimeout;
    window.addEventListener('scroll', function() {
      // Throttle scroll events for better performance
      if (scrollTimeout) {
        return;
      }
      scrollTimeout = setTimeout(function() {
        if (backToTopButton) {
          if (window.pageYOffset > 300) {
            backToTopButton.style.display = 'flex';
          } else {
            backToTopButton.style.display = 'none';
          }
        }
        scrollTimeout = null;
      }, 10);
    });
    
    // Smooth scroll to top when back to top button is clicked
    if (backToTopButton) {
      backToTopButton.addEventListener('click', function(e) {
        e.preventDefault();
        window.scrollTo({
          top: 0,
          behavior: 'smooth'
        });
        return false;
      });
    }

    // Highlight active navigation menu item based on scroll position - vanilla JS
    let navScrollTimeout;
    window.addEventListener('scroll', function() {
      // Throttle scroll events
      if (navScrollTimeout) {
        return;
      }
      navScrollTimeout = setTimeout(function() {
        const scrollPosition = window.pageYOffset;
        const navbarItems = document.querySelectorAll('.navbar-item');
        
        navbarItems.forEach(function(item) {
          const href = item.getAttribute('href');
          if (href && href.startsWith('#')) {
            const target = document.querySelector(href);
            if (target) {
              const targetTop = target.getBoundingClientRect().top + window.pageYOffset;
              const targetHeight = target.offsetHeight;
              
              if (targetTop <= scrollPosition + 200 && 
                  targetTop + targetHeight > scrollPosition + 200) {
                navbarItems.forEach(function(navItem) {
                  navItem.classList.remove('has-text-weight-bold');
                });
                item.classList.add('has-text-weight-bold');
              }
            }
          }
        });
        navScrollTimeout = null;
      }, 50);
    });

    // Smooth scroll for anchor links - vanilla JS
    const navbarItems = document.querySelectorAll('.navbar-item');
    navbarItems.forEach(function(item) {
      item.addEventListener('click', function(e) {
        const hash = this.hash;
        if (hash !== '') {
          e.preventDefault();
          const target = document.querySelector(hash);
          if (target) {
            const targetPosition = target.getBoundingClientRect().top + window.pageYOffset - 70;
            window.scrollTo({
              top: targetPosition,
              behavior: 'smooth'
            });
          }
        }
      });
    });

    // Initialize carousels after libraries are loaded (since scripts use defer)
    // Wait for bulmaCarousel to be available
    function waitForCarousel() {
      if (typeof bulmaCarousel !== 'undefined') {
        setTimeout(function() {
          initializeCarousels();
        }, 100);
      } else {
        setTimeout(waitForCarousel, 50);
      }
    }
    waitForCarousel();

    // Add fade-in animation to images on scroll
    const fadeInElements = document.querySelectorAll('.container img');
    
    const fadeInOptions = {
      threshold: 0.1,
      rootMargin: "0px 0px -100px 0px"
    };
    
    const fadeInObserver = new IntersectionObserver(function(entries, observer) {
      entries.forEach(entry => {
        if (!entry.isIntersecting) {
          return;
        }
        entry.target.style.opacity = "1";
        observer.unobserve(entry.target);
      });
    }, fadeInOptions);
    
    fadeInElements.forEach(image => {
      image.style.opacity = "0";
      image.style.transition = "opacity 0.8s ease-in-out";
      fadeInObserver.observe(image);
    });
});

// Function to initialize all carousels
function initializeCarousels() {
  console.log('Initializing carousels...');
  
  // Initialize generation carousel with specific options
  var generationCarouselOptions = {
    slidesToScroll: 1,
    slidesToShow: 1,
    loop: true,
    infinite: true,
    autoplay: false,
    pagination: true,
    navigationSwipe: true,
    breakpoints: [
      { changePoint: 480, slidesToShow: 1, slidesToScroll: 1 },
      { changePoint: 768, slidesToShow: 1, slidesToScroll: 1 }
    ]
  };

  // Initialize all carousels with standard options
  var carouselOptions = {
    slidesToScroll: 1,
    slidesToShow: 1,
    loop: true,
    infinite: true,
    autoplay: false,
    pagination: true,
    navigationSwipe: true,
    breakpoints: [
      { changePoint: 480, slidesToShow: 1, slidesToScroll: 1 },
      { changePoint: 768, slidesToShow: 1, slidesToScroll: 1 }
    ]
  };

  // Make sure elements exist before initializing
  var carousels = [];
  var generationCarousel = [];
  
  try {
    // Safe initialization - check if elements exist
    if (document.querySelectorAll('.carousel:not(#generation-carousel)').length > 0) {
      console.log('Initializing regular carousels');
      carousels = bulmaCarousel.attach('.carousel:not(#generation-carousel)', carouselOptions);
    }
    
    // Check if generation carousel exists
    if (document.getElementById('generation-carousel')) {
      console.log('Initializing generation carousel');
      generationCarousel = bulmaCarousel.attach('#generation-carousel', generationCarouselOptions);
    }
    
    // Add click events to custom navigation buttons - make it more robust
    document.querySelectorAll('.carousel-nav-left').forEach(function(navButton) {
      navButton.addEventListener('click', function() {
        // Find the closest carousel
        const carouselContainer = navButton.closest('.carousel-container');
        if (!carouselContainer) return;
        
        const carousel = carouselContainer.querySelector('.carousel');
        if (!carousel) return;
        
        // Find the carousel instance
        let carouselInstance = null;
        
        if (carousel.id === 'generation-carousel' && generationCarousel.length > 0) {
          carouselInstance = generationCarousel[0];
        } else {
          // Find in regular carousels
          carouselInstance = carousels.find(c => c && c.element === carousel);
        }
        
        if (carouselInstance) {
          carouselInstance.previous();
        }
      });
    });

    document.querySelectorAll('.carousel-nav-right').forEach(function(navButton) {
      navButton.addEventListener('click', function() {
        // Find the closest carousel
        const carouselContainer = navButton.closest('.carousel-container');
        if (!carouselContainer) return;
        
        const carousel = carouselContainer.querySelector('.carousel');
        if (!carousel) return;
        
        // Find the carousel instance
        let carouselInstance = null;
        
        if (carousel.id === 'generation-carousel' && generationCarousel.length > 0) {
          carouselInstance = generationCarousel[0];
        } else {
          // Find in regular carousels
          carouselInstance = carousels.find(c => c && c.element === carousel);
        }
        
        if (carouselInstance) {
          carouselInstance.next();
        }
      });
    });

    // Loop on each carousel initialized - check length first
    if (carousels.length > 0) {
      for(var i = 0; i < carousels.length; i++) {
        if (carousels[i]) {
          carousels[i].on('before:show', state => {
            console.log(state);
          });
        }
      }
    }
  } catch (error) {
    console.error('Error initializing carousels:', error);
  }
}
