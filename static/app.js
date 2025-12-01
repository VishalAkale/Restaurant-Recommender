// autosuggest for cuisine & client-side validation
const cuisineInput = document.getElementById('cuisineInput');
const suggestionsBox = document.getElementById('suggestions');
let timer = null;

cuisineInput.addEventListener('input', function(e){
  const q = this.value.trim();
  if (timer) clearTimeout(timer);
  timer = setTimeout(()=> fetchSuggestions(q), 220);
});

function fetchSuggestions(q){
  const url = SUGGEST_URL + '?q=' + encodeURIComponent(q || '');
  fetch(url).then(r=>r.json()).then(arr=>{
    suggestionsBox.innerHTML = '';
    if (!arr || arr.length === 0) return;
    arr.forEach(item=>{
      const div = document.createElement('div');
      div.className = 'item';
      div.textContent = item;
      div.onclick = () => {
        cuisineInput.value = item;
        suggestionsBox.innerHTML = '';
      };
      suggestionsBox.appendChild(div);
    });
  }).catch(err=>{
    console.error('suggestion err', err);
  });
}

// simple validation: disallow digits in city & cuisine on submit
const searchForm = document.getElementById('searchForm');
searchForm.addEventListener('submit', function(e){
  const city = (document.getElementById('cityInput')||{}).value || '';
  const cuisine = (document.getElementById('cuisineInput')||{}).value || '';
  const digitRegex = /\d/;
  if (digitRegex.test(city) || digitRegex.test(cuisine)){
    e.preventDefault();
    alert('Please remove digits from City or Cuisine fields.');
    return false;
  }
});
