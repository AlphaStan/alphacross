// Init variables and event handlers
var gameBoard = document.querySelector('#board');
gameBoard.addEventListener('click', getColumnId);
var resetButton = document.getElementById("reset_button");
resetButton.addEventListener('click', sendResetSignal);
var gameIsFinished = false;
var activeAI = false;
var activateAIButton = document.getElementById("activate_ai");
activateAIButton.addEventListener('click', activateAI);


if (performance.navigation.type == 1) {
    console.info( "Reset page on reload" );
    resetVariables();
}


function sendActionToFlask(column_id){
    if (gameIsFinished) return;
    var urlToPost = '/' + column_id + '/' + activeAI;
    $.ajax({
        url: urlToPost,
        type: 'GET',
        success: handleGridUpdate,
        error: defaultError
    });
}


function getColumnId(e){
    if (e.target.tagName !== 'BUTTON') return;

    var targetCell = e.target.parentElement;
    var targetRow = targetCell.parentElement;
    var targetRowCells = [...targetRow.children];

    var column_id = targetRowCells.indexOf(targetCell);

    sendActionToFlask(column_id)
}


function sendResetSignal(s){
    var urlToReset = '/reset';
    $.ajax({
        url: urlToReset,
        type: 'GET',
        success: resetBoard,
        error: defaultError
    });
}


function activateAI(s){
    activeAI = !activeAI;
    if (activeAI){
        activateAIButton.classList.add("active");
        activateAIButton.innerHTML = "AI ACTIVE";
        console.log("activate AI");
    }
    else {
        activateAIButton.classList.remove("active");
        activateAIButton.innerHTML = "AI INACTIVE";
        console.log("deactivate AI");
    }
}


function resetVariables(){
    activeAI = false;
    var urlToReset = '/reset';
    $.ajax({
        url: urlToReset,
        type: 'GET',
        success: resetBoard,
        error: defaultError
    });
}
