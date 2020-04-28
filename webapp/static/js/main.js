function reset_board(result, status, xhr){
    console.log("reset board");
    // Reset buttons class
    var buttons = board.getElementsByTagName('button');
    var button;
    for (var k in buttons){
        button = buttons[k];
        try{
            button.classList.remove(...button.classList);
        }
        catch(error){
            console.log('Cannot access element ' + k + ' of buttons: ' + error);
        }
    }
    document.getElementById('msg').innerHTML = '';
    gameIsFinished = false;
}


function default_error(xhr, status, error) {
    console.log(xhr.status + ": " + xhr.responseText);
}

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
        success: function(result, status, xhr){
            var parsed_result = $.parseJSON(result);
            var buttonClass = (parsed_result["player_id"] == 1) ? "red" : "yellow";
            var row = document.querySelector('tr:nth-child(' + (1 + parsed_result["row_id"]) + ')');
            var cell = row.querySelector('td:nth-child(' + (1 + parsed_result["col_id"]) + ')');
            if (parsed_result["column_is_full"]) {
                document.getElementById('msg').innerHTML = 'Sorry, column ' + (1 + parsed_result["col_id"]) + ' is full';
            }
            else{
                cell.firstElementChild.classList.add(buttonClass);
                document.getElementById('msg').innerHTML = '';
            }
            if (parsed_result["has_won"]) {
                if (activeAI){
                    document.getElementById('msg').innerHTML = 'You have won!';
                }
                else{
                    document.getElementById('msg').innerHTML = 'Player ' + parsed_result["player_id"] + ' has won!';
                }
                gameIsFinished = true;
            }
            if (activeAI & !parsed_result["column_is_full"] & !parsed_result["has_won"]) {
                var agentButtonClass = (parsed_result["agent_id"] == 2) ? "yellow" : "red";
                var row = document.querySelector('tr:nth-child(' + (1 + parsed_result["agent_row_id"]) + ')');
                var cell = row.querySelector('td:nth-child(' + (1 + parsed_result["agent_col_id"]) + ')');
                cell.firstElementChild.classList.add(agentButtonClass);
                document.getElementById('msg').innerHTML = '';
                if (parsed_result["agent_has_won"]) {
                    document.getElementById('msg').innerHTML = 'You have lost!';
                    gameIsFinished = true;
                }
            }
        },
        error: default_error
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
        success: reset_board,
        error: default_error
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
        success: reset_board,
        error: default_error
    });
}
