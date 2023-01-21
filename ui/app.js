// let inputImage = document.getElementById("input--image");
// let displayImage = document.getElementById("display--image");

const inputImage = document.querySelector(".input--image");
const displayImage = document.querySelector(".display--image");
let navLinks = document.querySelectorAll("nav a");
const textInput = document.querySelector(".text");

//Displaying the image in the container
inputImage.addEventListener("change", function () {
  let file = inputImage.files[0];
  let reader = new FileReader();
  reader.onload = function (e) {
    displayImage.src = e.target.result;
    displayImage.style.display = "block";
  };
  reader.readAsDataURL(file);

  //Sending the image to the backend
  const formData = new FormData();

  formData.append("file", inputImage.files[0]);

  fetch("http://localhost:1001/get_caption", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.text())
    .then((data) => textInput.value = data.slice(7, -5))
    .catch((error) => console.error(error));
    
});

//Scrolling smoothly to the section
navLinks.forEach(function (link) {
  link.addEventListener("click", function (e) {
    e.preventDefault();
    let targetId = link.getAttribute("href");
    let target = document.querySelector(targetId);
    target.scrollIntoView({ behavior: "smooth" });
  });
});


