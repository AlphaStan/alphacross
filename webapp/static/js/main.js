var gameBoard = document.querySelector('#board');
gameBoard.addEventListener('click', getColumnId);
var resetButton = document.getElementById("reset_button");
resetButton.addEventListener('click', sendResetSignal)
var gameIsFinished = false;

function sendActionToFlask(column_id){
    if (gameIsFinished) return;
    var urlToPost = '/' + column_id;
    $.ajax({
        url: urlToPost,
        type: 'GET',
        success: function(result, status, xhr){
            var parsed_result = $.parseJSON(result);
            var buttonClass = (parsed_result["agent_id"] == 1) ? "red" : "yellow";
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
                document.getElementById('msg').innerHTML = 'Player ' + parsed_result["agent_id"] + ' has won!';
                gameIsFinished = true;
               }
            },
        error: function(xhr, status, error) {
            console.log(xhr.status + ": " + xhr.responseText);
        }
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
        success: function(){
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
        },
        error: function(xhr, status, error) {
            console.log(xhr.status + ": " + xhr.responseText);
        }
    });
}
