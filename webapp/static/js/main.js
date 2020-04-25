var gameBoard = document.querySelector('#board');
gameBoard.addEventListener('click', getColumnId)

function sendActionToFlask(column_id){
    var urlToPost = '/' + column_id;
    var urlToReset = '/reset';
    $.ajax({
        url: urlToPost,
        type: 'GET',
        success: function(result, status, xhr){
            var parsed_result = $.parseJSON(result);
            var buttonClass = (parsed_result["agent_id"] == 1) ? "red" : "black";
            var row = document.querySelector('tr:nth-child(' + (1 + parsed_result["row_id"]) + ')');
            var cell = row.querySelector('td:nth-child(' + (1 + parsed_result["col_id"]) + ')');
            if (parsed_result["column_is_full"]) {
                document.getElementById('msg').innerHTML = 'Sorry, column ' + (1 + parsed_result["col_id"]) + ' is full';
            }
            cell.firstElementChild.classList.add(buttonClass);
            if (parsed_result["has_won"]) {
                document.getElementById('msg').innerHTML = 'Player ' + parsed_result["agent_id"] + ' has won!';
                $.ajax({
                    url: urlToReset,
                    type: 'GET',
                    success: function(result, status, xhr){
                        console.log('reset');
                        var buttons = document.getElementsByTagName('button');
                        var button;
                        for (var k in buttons){
                            button = buttons[k];
                            button.classList.remove(...button.classList);
                        }
                    },
                    error: function(xhr, status, error) {
                        console.log(xhr.status + ": " + xhr.responseText);
                    }
                });
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
