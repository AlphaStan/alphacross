describe("Main functions", () => {
  beforeEach(function(){
    content = `
    <h1 id="title" class="title">AlphaCross</h1>
    <p id="intro" class="intro">Have fun against our AI !</p>
    <div class="reset" id="rese
        <button type="button" class="reset_button" id="reset_button">RESET</button>
        <button type="button" class="activate_ai" id="activate_ai">AI INACTIVE</button>
    </div>
    <table id="board" class="board">
      <tbody>
        <tr>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
        </tr>
        <tr>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
        </tr>
        <tr>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
        </tr>
        <tr>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
        </tr>
        <tr>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
        </tr>
        <tr>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
          <td><button type="button"></button></td>
        </tr>
      </tbody>
    </table>
    <p id="msg" class="msg"></p>
    `
  });

  it('simulate a click on a cell', () => {
    // GIVEN
    document.body.innerHTML = content;
    // WHEN
    var row = document.querySelector('tr:nth-child(6)');
    var cell = row.querySelector('td:nth-child(1)');
    cell.click();
  });
})
