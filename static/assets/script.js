// Camera popup

// document.getElementById("camera-toggle").addEventListener("click", () => {
//   document.getElementById("camera-overlay").style.display = "flex";
// });

// document.getElementById("close-camera").addEventListener("click", () => {
//   document.getElementById("camera-overlay").style.display = "none";
// });


// Question switching

var chartState = true;
let currentQuestion = 1;

let questionsList = [
  {
    questionNumber: 1,
    questionText: "When will animals undergo hibernation",
    type: "long",
    userAnswer: "",
  }
];

function updateQuestion(questionData) {
  if (questionData.type === "mcq") {
    document.getElementById("question-wrapper").innerHTML = `
    <h3 class="question-text">${questionData.questionText}</h3>
    <form id="mcqForm" class="radio-options">
      <label class="radio">
        <input type="radio" name="r" value="1" />
        <span>${questionData.options[0]}</span>
      </label>
      <label class="radio">
        <input type="radio" name="r" value="2" />
        <span>${questionData.options[1]}</span>
      </label>
      <label class="radio">
        <input type="radio" name="r" value="3" />
        <span>${questionData.options[2]}</span>
      </label>
      <label class="radio">
        <input type="radio" name="r" value="4" />
        <span>${questionData.options[3]}</span>
      </label>
    </form>`;
  } else if (questionData.type === "long") {
    document.getElementById("question-wrapper").innerHTML = `
    <h3 class="question-text">${questionData.questionText}</h3>
    <textarea name="qq1" id="q1" cols="30" rows="10"></textarea>
    `;
  } else {
    console.error("Wrong question type, check the questions JSON");
  }
}

updateQuestion(questionsList[currentQuestion - 1]);

document.getElementById("previous-btn").addEventListener("click", () => {
  console.log(currentQuestion);
  if (currentQuestion > 1) {
    currentQuestion -= 1;
  }

  chartState = false;
  updateQuestion(questionsList[currentQuestion - 1]);
});

document.getElementById("next-btn").addEventListener("click", () => {
  console.log(currentQuestion);
  if (currentQuestion < 5) {
    currentQuestion += 1;
  }

  chartState = false;
  updateQuestion(questionsList[currentQuestion - 1]);
});

$("input[type=radio][name=r]").change(function () {
  console.log(this.value);
});

// Chart

let chartConfig = {
  chart: {
    type: "donut",
    height: "auto",
  },
  legend: {
    position: "bottom",
    fontWeight: 600,
  },
};

var chart1 = {
  series: [55, 41, 17],
  colors: ["#32af5e", "#e74242", "#a5a5a5"],
  labels: ["Correct Answers", "Wrong Answers", "Not Answered"],
  ...chartConfig,
};

var chart2 = {
  series: [44, 55, 41, 17, 15],
  ...chartConfig,
};

if (chartState) {
  var chartOne = new ApexCharts(document.querySelector("#chart1"), chart1);
  chartOne.render();

  var chartTwo = new ApexCharts(document.querySelector("#chart2"), chart2);
  chartTwo.render();
}
